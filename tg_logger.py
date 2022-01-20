import telegram_handler
# import python_telegram_logger as ptl
import logging
import time
from colorlog import ColoredFormatter

tg_bot_token = '831964163:AAH7SoaoqWzWIcHaS3yfdmMu-H46hhtUaXw'
tg_chat_id = 1147194
ik_chat_id = 94616973
sun_group_id = -321681009


def set_logger(level=logging.WARNING, name='logger', telegram=False):
    """Return a logger with a default ColoredFormatter."""
    file_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(funcName)s - %(message)s")
    stream_formatter = ColoredFormatter(
        "%(asctime)s [%(log_color)s%(levelname)-8s%(reset)s] %(white)s%(message)s",
        datefmt=None,
        reset=True,
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red',
        }
    )

    logger = logging.getLogger(name)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(stream_formatter)
    log_handler = logging.FileHandler("fits_parse.log")
    log_handler.setFormatter(file_formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(log_handler)

    if telegram:
        tg_handler = telegram_handler.TelegramHandler(tg_bot_token, sun_group_id)
        tg_formatter = telegram_handler.MarkdownFormatter()
        tg_handler.setFormatter(tg_formatter)
        logger.addHandler(tg_handler)

    logger.setLevel(level)

    return logger

# logger = logging.getLogger('tg_test3')

# handler = tgh.TelegramHandler(tg_token, sun_group_id)
# formatter = tgh.HtmlFormatter()
# # handler = ptl.Handler(tg_token, [sun_group_id])
# # formatter = ptl.MarkdownFormatter()


# handler.setFormatter(formatter)
# logger.addHandler(handler)
# logger.setLevel(logging.INFO)


logger = set_logger(telegram=True)


def send_test():
    logger.info('TEST')
    time.sleep(1)
    logger.error('TEST')
    time.sleep(1)
    logger.warning('warn')

    try:
        10 / 0
    except Exception as e:
        logger.warning('Something not good, get Exception: {}'.format(e))
        time.sleep(1)
        logger.error('Something not good, get Exception: {}'.format(e))


send_test()
