from .admin_console import interface as admin_console
from .main_page import interface as main_page
from .configuration_page import config_page


AVAILABLE_PAGES = {
    "Main Page": main_page,
    "Admin console": admin_console,
    "Configuration": config_page,
}
