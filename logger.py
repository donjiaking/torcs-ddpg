import logging
import os


def get_logger(log_path):
    with open(log_path,"w") as file:
        file.write('[Log Created.]\n')

    logger = logging.getLogger()

    logger.setLevel(level=logging.DEBUG)

    file_handler = logging.FileHandler(log_path, mode='a', encoding='UTF-8')
    file_handler.setLevel(logging.DEBUG)
    
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)

    # console_handler = logging.StreamHandler()
    # console_handler.setLevel(logging.DEBUG)

    logger.addHandler(file_handler)
    # logger.addHandler(console_handler)

    return logger


#### Testing
if __name__ == "__main__":
    logger = get_logger(log_path='log.txt')
    logger.info("info")
    logger.error("error")
    logger.debug("debug")
    logger.warning("warning")
