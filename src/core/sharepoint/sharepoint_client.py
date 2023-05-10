import os.path
import tempfile
from typing import Any, Dict, Generator, Optional

import msal

#noinspection PyPackageRequirements
from office365.entity_collection import EntityCollection
#noinspection PyPackageRequirements
from office365.graph_client import GraphClient
#noinspection PyPackageRequirements
from office365.onedrive.driveitems.driveItem import DriveItem
#noinspection PyPackageRequirements
from office365.onedrive.drives.drive import Drive
from office365.runtime.queries.update_entity import UpdateEntityQuery


class SharepointClient:
    def __init__(self, client_id: str, client_secret: str, tenant_id: str, site: str):
        self.__client_id = client_id
        self.__client_secret = client_secret
        self.__tenant_id = tenant_id
        self.__site = site
        self.drive = self.__connect()

    def __connect(self) -> Drive:
        def acquire_token_func():
            """
            Acquire token via MSAL
            """
            authority_url = f"https://login.microsoftonline.com/{self.__tenant_id}"
            app = msal.ConfidentialClientApplication(
                authority=authority_url,
                client_id=self.__client_id,
                client_credential=self.__client_secret,
            )
            token = app.acquire_token_for_client(
                scopes=["https://graph.microsoft.com/.default"]
            )
            return token

        self.client = GraphClient(acquire_token_func)
        x: Drive = self.client.sites[self.__site].drive.get().execute_query()
        return x

    def iterate_drive(
            self, drive: Drive, max_depth: int = -1
    ) -> Generator[DriveItem, None, None]:
        """
        Method for iterating a drive object
        :param drive: to be iterated
        :param max_depth: maximal folder hierarchy depth that should be iterated
        :return: Generator of drive items
        """
        drive_items: EntityCollection = drive.root.children.get().execute_query()
        return self.iterate_collection(drive, drive_items, max_depth, 0)

    def iterate_collection(
            self,
            drive: Drive,
            di: EntityCollection,
            max_depth: int = -1,
            current_depth: int = 0,
    ) -> Generator[DriveItem, None, None]:
        """
        Method for iterating a entity collection that belongs to a given drive
        :param drive: to be iterated
        :param di: to be iterated
        :param max_depth: maximal folder hierarchy depth that should be iterated
        :param current_depth: current folder hierarchy depth
        :return: Generator of drive items
        """
        for drive_item in di:
            drive_item: DriveItem = drive_item
            yield drive_item
            if drive_item.is_folder and (max_depth <= 0 or current_depth < max_depth):
                children = drive.items[drive_item.id].children.get().execute_query()
                self.iterate_collection(drive, children, current_depth + 1, max_depth)

    def get_item_by_path(self, path: str) -> Optional[DriveItem]:
        """
        Method allowing to get a drive item for a given path
        :param path: for which drive item should be resolved
        :return: Drive item for given path
        """
        return self.drive.root.get_by_path(path).get().execute_query()

    def get_bytes(self, file_path: str) -> bytes:
        """
        Downloads and returns the given file as raw bytes. Does not work for directories.
        :param file_path: The relative server path pointing to the file to download
        :return: The contents of the given file as raw bytes
        """
        item = self.get_item_by_path(file_path)
        if item is None:
            raise Exception(f'File {file_path} cannot be retrieved or does not exist')
        elif not item.is_file:
            raise Exception(f'Item {file_path} is not a file and cannot be represented as bytes')

        result = item.get_content().execute_query()
        return result.value

    def download(self, path: str) -> str:
        """
        NOTE: office365 has a bug when downloading .json files!!! If file suffix is changed e.g. to .jsn it works fine

        Method allowing to download a file from sharepoint
        :param path: for which drive item should be downloaded
        :return: path to the downloaded file
        """
        item = self.get_item_by_path(path)
        if item is None:
            raise Exception(f"Unknown file {path}")

        base_name = os.path.basename(item.web_url)
        file_path = os.path.join(tempfile.gettempdir(), base_name)
        mode = "wb"
        if base_name.lower().endswith(".json"):
            mode = "w"
        with open(file_path, mode) as downloaded_file:
            item.download(downloaded_file).execute_query()
        return file_path

    def rename(self, item_path: str, new_name: str):
        item = self.get_item_by_path(item_path)
        item.set_property('name', new_name)
        query = UpdateEntityQuery(item)
        self.client.add_query(query).execute_query()

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

    def upload_file(
            self, path: str, file_path: str, target_file_name: Optional[str] = None
    ):
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
