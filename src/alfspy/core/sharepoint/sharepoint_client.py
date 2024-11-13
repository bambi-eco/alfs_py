import copy
import json
import os.path
import tempfile
from typing import Generator, Optional, TypeVar, Callable, Union, Type, Final, Iterable, cast

import msal

from office365.entity_collection import EntityCollection
from office365.graph_client import GraphClient
from office365.onedrive.driveitems.driveItem import DriveItem
from office365.onedrive.drives.drive import Drive
from office365.onedrive.sites.site import Site
from office365.runtime.client_object import ClientObject
from office365.runtime.client_object_collection import ClientObjectCollection
from office365.runtime.client_result import ClientResult
from office365.runtime.odata.request import ODataRequest
from office365.runtime.odata.v3.json_light_format import JsonLightFormat
from office365.runtime.queries.function import FunctionQuery
from office365.runtime.queries.service_operation import ServiceOperationQuery
from office365.runtime.queries.update_entity import UpdateEntityQuery


# fix office365 request processing that assumes that no user ever would download JSON files because that could never possibly happen
def _process_response(self, response, query):
        return_type = query.return_type
        if return_type is None:
            return

        if isinstance(return_type, ClientObject):
            return_type.clear_state()

        # just ignore the content type for client results
        if isinstance(return_type, ClientResult):
            return_type.set_property('__value', response.content)
        else:
            if response.headers.get('Content-Type', '').lower().split(';')[0] == 'application/json':
                json_format = copy.deepcopy(self.json_format)
                if isinstance(json_format, JsonLightFormat):
                    if isinstance(query, (ServiceOperationQuery, FunctionQuery)):
                        json_format.function = query.name

                self.map_json(response.json(), return_type, json_format)
ODataRequest.process_response = _process_response


class SharepointClient:
    _T1: Final[Type] = TypeVar('_T1')
    _T2: Final[Type] = TypeVar('_T2')

    _DEFAULT_URL_BASE: Final[str] = 'https://login.microsoftonline.com/'
    _DEFAULT_SCOPE: Final[str] = 'https://graph.microsoft.com/.default'

    _client_id: str
    _client_secret: str
    _tenant_id: str
    _site_str: str

    _connected: bool
    _client: GraphClient
    _site: Site
    _drive: Drive

    def __init__(self, client_id: str, client_secret: str, tenant_id: str, site: str,
                 url_base: str = _DEFAULT_URL_BASE, scope: str = _DEFAULT_SCOPE) -> None:
        """
        A client that abstracts and facilitates sharepoint access.
        :param client_id: The ID of the azure client to use.
        :param client_secret: The client secret of the azure client to use.
        :param tenant_id: The tenant ID associated with the azure client to use.
        :param site: The site string of the site to be targeted.
        """
        self._client_id = client_id
        self._client_secret = client_secret
        self._tenant_id = tenant_id
        self._site_str = site
        self._url_base = url_base
        self._scope = scope

        self._connected = False
        self._connect()

    def _get_token(self) -> dict:
        authority_url = f'{self._url_base}{self._tenant_id}'
        app = msal.ConfidentialClientApplication(authority=authority_url, client_id=self._client_id,
                                                 client_credential=self._client_secret)
        token = app.acquire_token_for_client(scopes=[self._scope])
        return token

    def _connect(self) -> None:
        if not self._connected:
            self._client = GraphClient(self._get_token)
            self._site = self._client.sites[self._site_str]
            self._drive = cast(Drive, self._site.drive.get().execute_query())
            self._connected = True

    def _iterate_collection(self, entities: Iterable[DriveItem], max_depth: int = -1, cur_depth: int = 0) -> Generator[DriveItem, None, None]:
        for drive_item in entities:
            drive_item: DriveItem = drive_item
            yield drive_item
            if drive_item.is_folder and (max_depth <= 0 or cur_depth < max_depth):
                children = cast(ClientObjectCollection, self._drive.items[drive_item.id].children.get().execute_query())
                self._iterate_collection(children, cur_depth + 1, max_depth)

    def iterate_collection(self, entities: Iterable[DriveItem], max_depth: int = -1) -> Generator[DriveItem, None, None]:
        """
        Method for iterating though all items in an entity collection associated with the connected drive.
        :param entities: The entity collection to be iterated.
        :param max_depth: The maximum depth to which the directories within the collection should be iterated recursively
         (defaults to -1). Any value smaller than one causes unlimited iteration.
        :return: Generator of drive items.
        """
        return self._iterate_collection(entities, max_depth, 0)

    def iterate_items(self, max_depth: int = -1) -> Generator[DriveItem, None, None]:
        """
        Method for iterating all items associated with the connected drive.
        :param max_depth: The maximum depth to which the drive directory is to be iterated (defaults to -1). Any value
        smaller than one causes unlimited iteration.
        :return: Generator of drive items.
        """
        drive_items = cast(EntityCollection, self._drive.root.children.get().execute_query())
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
        Retrieves a drive item by its identifier.
        :param identifier: The identifier associated with the item to retrieve.
        :return: If the id is invalid or retrieval was unsuccessful ``None``; otherwise the associated item.
        """
        return self._try_or_default(lambda: self._drive.items[identifier].get().execute_query())

    def get_item_by_path(self, item_path: str) -> Optional[DriveItem]:
        """
        Retrieves a drive item by its server relative path.
        :param item_path: The server relative path pointing at the item to retrieve.
        :return: If the path is invalid or retrieval was unsuccessful ``None``; otherwise the associated item.
        """
        return self._try_or_default(lambda: self._drive.root.get_by_path(item_path).get().execute_query())

    def item_exists_by_id(self, identifier: str) -> bool:
        """
        Checks whether an item with the given identifier exists on the Sharepoint.
        :param identifier: The identifier associated with the file to check.
        :return: Whether the item exists.
        """
        return self.get_item_by_id(identifier) is not None

    def item_exists_by_path(self, item_path: str) -> bool:
        """
        Checks whether an item with the given identifier exists on the Sharepoint.
        :param item_path: The path pointing at the position to check.
        :return: Whether the item exists.
        """
        return self.get_item_by_path(item_path) is not None

    def get_file_bytes(self, item: DriveItem) -> Optional[bytes]:
        """
        Downloads and returns the content of the given item as raw bytes. Does not work for directories.
        :param item: The drive item whose content to retrieve.
        :return: If there is an error retrieving the data ``None``; otherwise the contents of the given item as raw bytes.
        """
        return self._try_or_default(lambda: item.get_content().execute_query().value)

    def get_file_bytes_by_id(self, identifier: str) -> Optional[bytes]:
        """
        Downloads and returns the file associated with the given identifier as raw bytes. Does not work for directories.
        :param identifier: The identifier associated with the file to retrieve.
        :return: If there is an error retrieving the data ``None``; otherwise the contents of the given file as raw bytes.
        """
        return self.get_file_bytes(self.get_item_by_id(identifier))

    def get_file_bytes_by_path(self, file_path: str) -> Optional[bytes]:
        """
        Downloads and returns the file associated with the given server relative path as raw bytes. Does not work for directories.
        :param file_path: The server relative path pointing at the file to retrieve.
        :return: If there is an error retrieving the data ``None``; otherwise the contents of the given file as raw bytes.
        """
        return self.get_file_bytes(self.get_item_by_path(file_path))

    def download_item(self, item: DriveItem) -> Optional[str]:
        """
        Downloads the given item to a temporary directory and returns its local location.
        :param item: The item to be downloaded.
        :return: If the item was not downloaded successfully ``None``; otherwise the location to which the item has been
        downloaded to.
        """
        def func():
            item.get().execute_query()
            file_path = os.path.join(tempfile.gettempdir(), item.name)
            content = item.get_content().execute_query()
            content = content.value
            if isinstance(content, bytes):
                mode = "wb+"
            else:
                mode = "w+"
                if not isinstance(content, str):   # sometimes the value is directly parsed to an object
                    try:
                        content = json.dumps(content)
                    except (TypeError, OverflowError):
                        content = str(content)

            with open(file_path, mode) as downloaded_file:
                downloaded_file.write(content)
            return file_path
        return self._try_or_default(func, exception_types=tuple())

    def download_item_by_id(self, identifier: str) -> Optional[str]:
        """
        Downloads the item associated with the given identifier to a temporary directory and returns its local location.
        :param identifier: The identifier associated with the item to download.
        :return: If the item was not downloaded successfully ``None``; otherwise the location to which the item has been
        downloaded to.
        """
        item = self.get_item_by_id(identifier)
        return self.download_item(item)

    def download_item_by_path(self, item_path: str) -> Optional[str]:
        """
        Downloads the item associated with the given server relative path to a temporary directory and returns its local location.
        :param item_path: The server relative path pointing at the item to download.
        :return: If the item was not downloaded successfully ``None``; otherwise the location to which the item has been
        downloaded to.
        """
        item = self.get_item_by_path(item_path)
        return self.download_item(item)

    def rename_item(self, item: DriveItem, new_name: str) -> bool:
        """
        Renames the given item.
        :param item: The item to rename.
        :param new_name: The name to apply to the given item.
        :return: Whether the renaming process was successful.
        """
        def func():
            item.set_property('name', new_name)
            query = UpdateEntityQuery(item)
            self._client.add_query(query).execute_query()

        return self._try_for_success(func)

    def rename_item_by_id(self, identifier: str, new_name:str) -> bool:
        """
        Renames the item associated with the given identifier.
        :param identifier: The identifier associated with the item to rename.
        :param new_name: The name to apply to the given item.
        :return: Whether the renaming process was successful.
        """
        item = self.get_item_by_id(identifier)
        return self.rename_item(item, new_name)

    def rename_item_by_path(self, item_path: str, new_name: str) -> bool:
        """
        Renames the item associated with the given server relative path.
        :param item_path: The server relative path pointing at the item to rename.
        :param new_name: The name to apply to the given item.
        :return: Whether the renaming process was successful.
        """
        item = self.get_item_by_path(item_path)
        return self.rename_item(item, new_name)

    def get_children(self, item: DriveItem) -> list[DriveItem]:
        """
        Queries and returns a list of a child items within a given folder. Returns an empty list if the given path
        points at a non-dictionary item.
        :param item: The item whose children to return.
        :return: If the item could not be retrieved ``None``; otherwise a list of all direct children of the given item.
        """
        def func():
            items = []
            children = cast(ClientObjectCollection, item.children.get().execute_query())
            for child in children:
                items.append(child)
            return items

        return self._try_or_default(func)

    def get_children_by_id(self, identifier: str) -> list[DriveItem]:
        """
        Queries and returns a list of a child items within a given folder. Returns an empty list if the given path
        points at a non-dictionary item.
        :param identifier: The identifier associated with the item whose children to return.
        :return: If the item could not be retrieved ``None``; otherwise a list of all direct children of the given item.
        """
        item = self.get_item_by_id(identifier)
        return self.get_children(item)

    def get_children_by_path(self, item_path: str) -> list[DriveItem]:
        """
        Queries and returns a list of a child items within a given folder. Returns an empty list if the given path
        points at a non-dictionary item.
        :param item_path: The server relative path pointing at the item whose children to return.
        :return: If the item could not be retrieved ``None``; otherwise a list of all direct children of the given item.
        """
        item = self.get_item_by_path(item_path)
        return self.get_children(item)

    def upload_file(self, target_dir_path: str, file_path: str, overwrite: bool = False, chunk_size: int = 5242880,
                    progress_callback: Optional[Callable[[int], None]] = None) -> Optional[DriveItem]:
        """
        Uploads a file to given directory on the Sharepoint. The upload itself is carried out by segmenting the given
        data into chunks.
        :param target_dir_path: The server relative path of the directory the given file should be saved to.
        :param file_path: The local path pointing at the file to upload.
        :param overwrite: Whether to allow overwriting existing files (defaults to ``True``).
        :param chunk_size: The chunk size to be used for segmenting the data to upload (defaults to ``5242880``).
        :param progress_callback: A callback to be called when bytes are uploaded (optional). The first and only
        parameter represents the amount of uploaded bytes.
        :return: If the upload was not successful or any of the given paths are invalid ``None``; otherwise the item
        associated with the just uploaded file.
        """
        
        # https://github.com/vgrem/Office365-REST-Python-Client/blob/master/examples/onedrive/upload_large_file.py
        folder = self.get_item_by_path(target_dir_path)
        if folder is None or not folder.is_folder:
            raise Exception(f'Invalid target directory path: {target_dir_path}')

        if not os.path.exists(file_path) or not os.path.isfile(file_path):
            raise Exception(f'Invalid file path. Target does not exist or is not a file: {file_path}')

        if not overwrite:
            file_name = os.path.basename(file_path)
            to_be_file_path = os.path.join(target_dir_path, file_name)
            if self.item_exists_by_path(to_be_file_path):
                raise Exception(f'The target directory already contains a file named "{file_name}" and overwriting was specifically prohibited')

        def func():
            item = folder.resumable_upload(file_path, chunk_size=chunk_size, chunk_uploaded=progress_callback)
            item.execute_query()
            return item

        return self._try_or_default(func)

