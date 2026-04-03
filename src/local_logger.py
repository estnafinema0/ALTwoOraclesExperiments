import pathlib
import sys

class Logger:
    def __init__(self, verbose_console: bool, log_file: pathlib.Path | None = None):
        self.verbose_console = verbose_console
        self.log_file = log_file
        if not verbose_console and log_file is None:
            self.warn("Verbose logging is enabled, but log file is not provided so it won't be written")

    def debug(self, complete_message: str):
        print(f'[DEBUG]: {complete_message}')
        if self.log_file is not None:
            with self.log_file.open('a+') as file:
                print(f'[DEBUG]: {complete_message}', file=file)
    
    def info(self, complete_message: str):
        print(f'[INFO]: {complete_message}')
        if self.log_file is not None:
            with self.log_file.open('a+') as file:
                print(f'[INFO]: {complete_message}', file=file)
    
    def warn(self, complete_message: str, shortened_message: str | None = None):
        if self.verbose_console:
            print(f'[WARNING]: {complete_message}', file=sys.stderr)
        else:
            if shortened_message is not None:
                print(f'[WARNING]: {shortened_message}', file=sys.stderr)
            else:
                print(f'[WARNING]: {complete_message}', file=sys.stderr)
        if self.log_file is not None:
            with self.log_file.open('a+') as file:
                print(f'[WARNING]: {complete_message}', file=file)

    def error(self, complete_message: str, shortened_message: str | None = None):
        if self.verbose_console:
            print(f'[ERROR]: {complete_message}', file=sys.stderr)
        else:
            if shortened_message is not None:
                print(f'[ERROR]: {shortened_message}', file=sys.stderr)
            else:
                print(f'[ERROR]: {complete_message}', file=sys.stderr)
        if self.log_file is not None:
            with self.log_file.open('a+') as file:
                print(f'[ERROR]: {complete_message}', file=file)

