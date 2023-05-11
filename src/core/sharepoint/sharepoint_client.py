import os.path
import tempfile
from typing import Any, Dict, Generator, Optional, TypeVar, Callable, Union, Type

import msal
# noinspection PyPackageRequirements
from office365.entity_collection import EntityCollection
# noinspection PyPackageRequirements
from office365.graph_client import GraphClient
# noinspection PyPackageRequirements
from office365.onedrive.driveitems.driveItem import DriveItem
# noinspection PyPackageRequirements
from office365.onedrive.drives.drive import Drive
# noinspection PyPackageRequirements
from office365.onedrive.sites.site import Site
# noinspection PyPackageRequirements
from office365.runtime.queries.update_entity import UpdateEntityQuery


class SharepointClient:
    _client_id: str
    _client_secret: str
    _tenant_id: str
    _site_str: str

    _connected: bool
    _client: GraphClient
    _site: Site
    _drive: Drive

    _T1 = TypeVar('_T1', covariant=True)
    _T2 = TypeVar('_T2', covariant=True)

    def __init__(self, client_id: str, client_secret: str, tenant_id: str, site: str):
        """
        A client that abstracts and facilitates sharepoint access.
        :param client_id: The ID of the azure client to use
        :param client_secret: The client secret of the azure client to use
        :param tenant_id: The tenant ID associated with the azure client to use
        :param site: The site string of the site to be targeted
        """
        self._client_id = client_id
        self._client_secret = client_secret
        self._tenant_id = tenant_id
        self._site_str = site
        self._connected = False

    def _get_token(self) -> dict:
        authority_url = f"https://login.microsoftonline.com/{self._tenant_id}"
        app = msal.ConfidentialClientApplication(
            authority=authority_url,
            client_id=self._client_id,
            client_credential=self._client_secret,
        )
        token = app.acquire_token_for_client(
            scopes=["https://graph.microsoft.com/.default"]
        )
        return token

    def _connect(self) -> None:
        if not self._connected:
            self._client = GraphClient(self._get_token)
            self._site = self._client.sites[self._site_str]
            self._drive = self._site.drive.get().execute_query()
            self._connected = True

    def _iterate_collection(self, entities: EntityCollection, max_depth: int = -1, cur_depth: int = 0):
        for drive_item in entities:
            drive_item: DriveItem = drive_item
            yield drive_item
            if drive_item.is_folder and (max_depth <= 0 or cur_depth < max_depth):
                children = self._drive.items[drive_item.id].children.get().execute_query()
                self._iterate_collection(children, cur_depth + 1, max_depth)

    def iterate_collection(self, entities: EntityCollection, max_depth: int = -1) -> Generator[DriveItem, None, None]:
        """
        Method for iterating though all items in an entity collection associated with the connected drive
        :param entities: The entity collection to be iterated
        :param max_depth: The maximum depth to which the directories within the collection should be iterated recursively
         (defaults to -1). Any value smaller than one causes unlimited iteration.
        :return: Generator of drive items
        """
        return self._iterate_collection(entities, max_depth, 0)

    def iterate_items(self, max_depth: int = -1) -> Generator[DriveItem, None, None]:
        """
        Method for iterating all items associated with the connected drive
        :param max_depth: The maximum depth to which the drive directory is to be iterated (defaults to -1). Any value
        smaller than one causes unlimited iteration.
        :return: Generator of drive items
        """
        drive_items: EntityCollection = self._drive.root.children.get().execute_query()
        return self.iterate_collection(drive_items, max_depth)

    @staticmethod
    def _try_or_default(func: Callable[[], _T1], default: _T2 = None,
                        exception_types: tuple[Type[Exception]] = (Exception, )) -> Union[_T1, _T2]:
        try:
            return func()
        except exception_types:
            return default

    @staticmethod
    def _try_for_success(func: Callable[[], None], exception_types: tuple[Type[Exception]] = (Exception, )) -> bool:
        try:
            func()
        except exception_types:
            return False
        return True

    def get_item_by_id(self, identifier: str)-> Optional[DriveItem]:
        """
        Retrieves a drive item by its identifier
        :param identifier: The identifier associated with the item to retrieve
        :return: If the id is invalid or retrieval was unsuccessful ``None``; otherwise the associated item
        """
        return self._try_or_default(lambda: self._drive.items[identifier].children.get().execute_query())

    def get_item_by_path(self, path: str) -> Optional[DriveItem]:
        """
        Retrieves a drive item by its server relative path
        :param path: The server relative path pointing at the item to retrieve
        :return: If the path is invalid or retrieval was unsuccessful ``None``; otherwise the associated item
        """
        return self._try_or_default(self._drive.root.get_by_path(path).get().execute_query())

    def get_file_bytes(self, item: DriveItem) -> Optional[bytes]:
        """
        Downloads and returns the content of the given item as raw bytes. Does not work for directories.
        !!! Does not work correctly for files with a '.json' extension due to a bug within the Office 365 package !!!
        :param item: The drive item whose content to retrieve
        :return: If there is an error retrieving the data ``None``; otherwise the contents of the given item as raw bytes
        """
        return self._try_or_default(item.get_content().execute_query().value)

    def get_file_bytes_by_id(self, file_id: str) -> Optional[bytes]:
        """
        Downloads and returns the file associated with the given identifier as raw bytes. Does not work for directories.
        !!! Does not work correctly for files with a '.json' extension due to a bug within the Office 365 package !!!
        :param file_id: The identifier associated with the file to retrieve
        :return: If there is an error retrieving the data ``None``; otherwise the contents of the given file as raw bytes
        """
        return self.get_file_bytes(self.get_item_by_id(file_id))

    def get_file_bytes_by_path(self, file_path: str) -> Optional[bytes]:
        """
        Downloads and returns the file associated with the given server relative path as raw bytes. Does not work for directories.
        !!! Does not work correctly for files with a '.json' extension due to a bug within the Office 365 package !!!
        :param file_path: The server relative path pointing at the file to retrieve
        :return: If there is an error retrieving the data ``None``; otherwise the contents of the given file as raw bytes
        """
        return self.get_file_bytes(self.get_item_by_path(file_path))

    def download_item(self, item: DriveItem) -> Optional[str]:
        """
        Downloads the given item to a temporary directory and returns its local location.
        !!! Does not work correctly for files with a '.json' extension due to a bug within the Office 365 package !!!
        :param item: The item to be downloaded
        :return: If the item was not downloaded successfully ``None``; otherwise the location to which the item has been
        downloaded to
        """
        def func():
            base_name = os.path.basename(item.web_url)
            file_path = os.path.join(tempfile.gettempdir(), base_name)
            mode = "wb"
            if base_name.lower().endswith(".json"):
                mode = "w"
            with open(file_path, mode) as downloaded_file:
                item.download(downloaded_file).execute_query()
            return file_path
        return self._try_or_default(func)

    def download_item_by_id(self, identifier: str) -> Optional[str]:
        """
        Downloads the item associated with the given identifier to a temporary directory and returns its local location.
        !!! Does not work correctly for files with a '.json' extension due to a bug within the Office 365 package !!!
        :param identifier: The identifier associated with the item to download
        :return: If the item was not downloaded successfully ``None``; otherwise the location to which the item has been
        downloaded to
        """
        item = self.get_item_by_id(identifier)
        return self.download_item(item)

    def download_item_by_path(self, path: str) -> Optional[str]:
        """
        Downloads the item associated with the given server relative path to a temporary directory and returns its local location.
        !!! Does not work correctly for files with a '.json' extension due to a bug within the Office 365 package !!!
        :param path: The server relative path pointing at the item to download
        :return: If the item was not downloaded successfully ``None``; otherwise the location to which the item has been
        downloaded to
        """
        item = self.get_item_by_path(path)
        return self.download_item(item)

    def rename_item(self, item: DriveItem, new_name: str) -> bool:
        """
        Renames the given item.
        :param item: The item to rename
        :param new_name: The name to apply to the given item
        :return: Whether the renaming process was successful
        """
        def func():
            item.set_property('name', new_name)
            query = UpdateEntityQuery(item)
            self._client.add_query(query).execute_query()

        return self._try_for_success(func)

    def rename_item_by_id(self, identifier: str, new_name:str) -> bool:
        """
        Renames the item associated with the given identifier
        :param identifier: The identifier associated with the item to rename
        :param new_name: The name to apply to the given item
        :return: Whether the renaming process was successful
        """
        item = self.get_item_by_id(identifier)
        return self.rename_item(item, new_name)

    def rename_item_by_path(self, item_path: str, new_name: str) -> bool:
        """
        Renames the item associated with the given server relative path
        :param item_path: The server relative path pointing at the item to rename
        :param new_name: The name to apply to the given item
        :return: Whether the renaming process was successful
        """
        item = self.get_item_by_path(item_path)
        return self.rename_item(item, new_name)



    def get_content(self, path: str) -> Dict[str, Any]:
        """
        Method for getting the content of a given sharepoint folder
        :param path: of folder for which content should be retrieved
        :return: dictionary describing all items
        """
        res = {}
        item = self.get_item_by_path(path)
        if item is None:
            return res
        children = item.children.get().execute_query()

        items = []
        for child in children:
            inner_dict = {
                "name": child.name,
                "isDir": child.is_folder,
                "webUrl": child.web_url,
            }
            items.append(inner_dict)
        res["items"] = items
        return res

    def upload_file(self, path: str, file_path: str, target_file_name: Optional[str] = None):
        # https://github.com/vgrem/Office365-REST-Python-Client/blob/master/examples/onedrive/upload_large_file.py
        folder = self.get_item_by_path(path)
        if folder is None or not folder.is_folder:
            raise Exception(f"Unknown folder for path: {path}")
        file_path = os.path.normpath(file_path)
        if not os.path.exists(file_path) or not os.path.isfile(file_path):
            raise Exception(f"Unknown file for path: {file_path}")
        chunk_size = 5 * 1024 * 1024

        def print_progress(range_pos):
            print("{0} bytes uploaded".format(range_pos))

        res = (
            folder.resumable_upload(
                os.path.normpath(file_path),
                chunk_size=chunk_size,
                chunk_uploaded=print_progress,
            )
            .get()
            .execute_query()
        )
        return res
