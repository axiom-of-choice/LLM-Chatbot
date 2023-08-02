# Firts we need to initialize the config package to import pages
from .config import *
from pages import main_page, admin_console, config_page

AVAILABLE_PAGES = {
    "Main Page": main_page,
    "Admin console": admin_console,
    "Configuration": config_page,
}
