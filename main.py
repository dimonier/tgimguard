import asyncio
import logging
import logging.handlers
import os
from typing import List

from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart
from aiogram.types import Message
from aiogram.enums import ChatType
from aiogram.client.default import DefaultBotProperties
from aiogram import Router
from dotenv import load_dotenv

# AICODE-NOTE: [CONTEXT] Minimal single-file bot to moderate images by OCR via OpenAI API


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
def _setup_file_logging() -> None:
    """Configure daily rotating file logging and keep 30 days."""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "tgimguard.log")
    file_handler = logging.handlers.TimedRotatingFileHandler(
        log_path, when="midnight", backupCount=30, encoding="utf-8"
    )
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s() - %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    logging.getLogger().addHandler(file_handler)

_setup_file_logging()

# Load .env early
load_dotenv()

router = Router()


def get_env(name: str) -> str:
    """Get required environment variable or raise."""
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required env var: {name}")
    return value


OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL_ID = os.getenv("OPENAI_MODEL_ID", "gpt-4o-mini")
PROHIBITED_SUBSTRINGS: List[str] = [s for s in os.getenv("PROHIBITED_STRINGS", "").split("|") if s]
CHAT_OWNER_ID = os.getenv("CHAT_OWNER_ID")  # optional, but recommended
IMAGE_COMPRESSION_THRESHOLD_KB = int(os.getenv("IMAGE_COMPRESSION_THRESHOLD_KB", "300"))
IMAGE_RESIZE_WIDTH_PX = int(os.getenv("IMAGE_RESIZE_WIDTH_PX", "800"))


async def get_chat_owner_id(bot: Bot, chat_id: int) -> int | None:
    """Return creator user id for chat, or None if not found."""
    try:
        admins = await bot.get_chat_administrators(chat_id)
    except Exception as e:
        logger.warning("Failed to fetch chat administrators: %s", e)
        return None
    for admin in admins:
        # Aiogram v3 represents owner as status "creator"
        if getattr(admin, "status", None) == "creator":
            return admin.user.id
    return None


@router.message(CommandStart())
async def start(message: Message) -> None:
    """Reply with brief info."""
    await message.answer("Bot is active. Send an image in a group to be moderated.")


def _detect_mime_type(image_bytes: bytes) -> str:
    """Return best-effort MIME type for given image bytes.

    Checks magic numbers of common formats; falls back to image/jpeg if unknown.
    """
    header = image_bytes[:12]
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if header[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if header.startswith(b"GIF87a") or header.startswith(b"GIF89a"):
        return "image/gif"
    if header.startswith(b"RIFF") and header[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"


def _prepare_image_for_llm(image_bytes: bytes) -> tuple[bytes, str]:
    """Resize to configured width and convert to JPEG if size exceeds threshold.

    Returns (processed_bytes, mime_type).
    """
    threshold_bytes = max(0, IMAGE_COMPRESSION_THRESHOLD_KB) * 1024
    if threshold_bytes and len(image_bytes) > threshold_bytes:
        try:
            import io
            from PIL import Image  # type: ignore

            with Image.open(io.BytesIO(image_bytes)) as img:
                # Convert to RGB for JPEG compatibility
                if img.mode not in ("RGB", "L"):
                    img = img.convert("RGB")

                width, height = img.size
                target_w = int(IMAGE_RESIZE_WIDTH_PX)
                if width > target_w and target_w > 0:
                    new_h = int(height * target_w / max(1, width))
                    img = img.resize((target_w, new_h), Image.LANCZOS)

                out = io.BytesIO()
                img.save(out, format="JPEG")
                processed = out.getvalue()
                logger.info(
                    "Image re-saved: new_size=%d bytes (old_size=%d bytes)",
                    len(processed),
                    len(image_bytes),
                )
                return processed, "image/jpeg"
        except Exception as e:
            logger.warning("Image preprocessing failed, using original: %s", e)

    # Below threshold or processing failed: keep original and detect mime
    return image_bytes, _detect_mime_type(image_bytes)


async def extract_text_via_openai(image_bytes: bytes) -> str:
    """Call OpenAI-compatible API with image to extract visible text via OCR.
    Uses multimodal prompt with input_image content type.
    """
    # Lazy import to keep surface minimal
    from openai import OpenAI

    client = OpenAI(base_url=OPENAI_BASE_URL, api_key=get_env("OPENAI_API_KEY"))

    # Preprocess image per limits and upload as data URL
    # For small images this is fine; for large uploads consider hosting.
    import base64

    processed_bytes, mime_type = _prepare_image_for_llm(image_bytes)
    b64 = base64.b64encode(processed_bytes).decode("ascii")
    data_url = f"data:{mime_type};base64,{b64}"

    system_prompt = (
        "You are an OCR engine. Extract ONLY readable text from the image. "
        "Return plain text without explanations."
    )

    resp = client.chat.completions.create(
        model=OPENAI_MODEL_ID,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract visible text from this image."},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
        temperature=0,
    )

    text = resp.choices[0].message.content or ""
    return text.strip()


def contains_prohibited(text: str) -> bool:
    """Return True if any prohibited substring is in text (case-insensitive)."""
    if not PROHIBITED_SUBSTRINGS:
        return False
    lowered = text.lower()
    return any(s.lower() in lowered for s in PROHIBITED_SUBSTRINGS)


def _format_user_info(user) -> str:
    """Build a single-line string combining available user identifiers."""
    parts: list[str] = []
    username = getattr(user, "username", None)
    if username:
        uname = str(username)
        parts.append(uname if uname.startswith("@") else f"@{uname}")

    first_name = getattr(user, "first_name", None)
    last_name = getattr(user, "last_name", None)
    full_name = " ".join([n for n in [first_name, last_name] if n])
    if full_name:
        parts.append(full_name)

    user_id = getattr(user, "id", None)
    if user_id is not None:
        parts.append(f"id={user_id}")

    return " | ".join(parts) or "unknown user"


def _format_chat_info(chat) -> str:
    """Build a single-line string combining available chat identifiers."""
    parts: list[str] = []
    title = getattr(chat, "title", None)
    if title:
        parts.append(str(title))

    username = getattr(chat, "username", None)
    if username:
        uname = str(username)
        parts.append(uname if uname.startswith("@") else f"@{uname}")

    chat_type = getattr(chat, "type", None)
    if chat_type:
        parts.append(f"type={chat_type}")

    chat_id = getattr(chat, "id", None)
    if chat_id is not None:
        parts.append(f"id={chat_id}")

    return " | ".join(parts) or "unknown chat"


@router.message(F.content_type.in_({"photo", "document"}))
async def on_image(message: Message, bot: Bot) -> None:
    """Handle images sent to group/supergroup chats."""
    if message.chat.type not in {ChatType.GROUP, ChatType.SUPERGROUP}:
        return

    file_id = None
    if message.photo:
        file_id = message.photo[-1].file_id
    elif message.document and (message.document.mime_type or "").startswith("image/"):
        file_id = message.document.file_id

    if not file_id:
        return

    try:
        file = await bot.get_file(file_id)
        file_path = file.file_path
        # Download bytes
        from aiohttp import ClientSession

        tg_file_url = f"https://api.telegram.org/file/bot{bot.token}/{file_path}"
        async with ClientSession() as session:
            async with session.get(tg_file_url) as resp:
                resp.raise_for_status()
                image_bytes = await resp.read()

        text = await extract_text_via_openai(image_bytes)
        logger.info("OCR extracted %d chars", len(text))
        preview = " ".join(text.split())[:50]
        logger.info("OCR text preview: %s", preview)

        is_prohibited = contains_prohibited(text)
        logger.info("Image evaluation: %s", "prohibited" if is_prohibited else "OK")

        if is_prohibited:
            # Delete message
            try:
                await bot.delete_message(chat_id=message.chat.id, message_id=message.message_id)
            except Exception as e:
                logger.warning("Failed to delete message: %s", e)

            # Ban user
            try:
                await bot.ban_chat_member(chat_id=message.chat.id, user_id=message.from_user.id)
            except Exception as e:
                logger.warning("Failed to ban user: %s", e)

            # Notify owner: env first, otherwise autodetect creator
            owner_id = int(CHAT_OWNER_ID) if CHAT_OWNER_ID else await get_chat_owner_id(bot, message.chat.id)
            if owner_id:
                try:
                    user_info = _format_user_info(message.from_user)
                    chat_info = _format_chat_info(message.chat)
                    await bot.send_message(
                        chat_id=owner_id,
                        text=(
                            f"User {user_info} banned in chat {chat_info} "
                            f"for the image containing the following text:"
                            f"\n\n{text}"
                        ),
                    )
                except Exception as e:
                    logger.warning("Failed to notify owner: %s", e)
    except Exception as e:
        logger.exception("Image handling error: %s", e)


async def main() -> None:
    token = get_env("TELEGRAM_BOT_TOKEN")

    dp = Dispatcher()
    dp.include_router(router)

    bot = Bot(token=token, default=DefaultBotProperties(parse_mode=None))

    await dp.start_polling(bot)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        pass
