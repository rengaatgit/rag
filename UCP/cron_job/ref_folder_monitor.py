import time
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from config import Config
from data_processor.file_processor import process_files

class RefFolderHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if not event.is_directory:
            process_files(Config.REF_FOLDER, "ucp600_rules")

def run_monitor():
    event_handler = RefFolderHandler()
    observer = Observer()
    observer.schedule(event_handler, Config.REF_FOLDER, recursive=False)
    observer.start()
    try:
        while True:
            time.sleep(1800)  # 30 minutes
    except KeyboardInterrupt:
        observer.stop()
    observer.join()