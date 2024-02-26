import dropbox
import os

class Dropbox():
    
    def __init__(self, token):
        self.token = token
        self.dbx = dropbox.Dropbox(token)
        try:
            self.dbx.users_get_current_account()
        except:
            raise ValueError("invalid token")


    def upload_file(self, local_file_path, dropbox_path):
        with open(local_file_path, 'rb') as f:
            self.dbx.files_upload(f.read(), dropbox_path, mode=dropbox.files.WriteMode("overwrite"))

    def upload_folder(self, local_folder_path, dropbox_folder_path, ext_forbidden=None):
        for root, dirs, files in os.walk(local_folder_path):
            for filename in files:

                if ext_forbidden is not None:
                    _, ext = os.path.splitext(filename)
                    if ext in ext_forbidden:
                        continue

                local_path = os.path.join(root, filename)
                relative_path = os.path.relpath(local_path, local_folder_path)
                dropbox_path = os.path.join(dropbox_folder_path, relative_path).replace(os.sep, '/')
                self.upload_file(local_path, dropbox_path)
                print(f"Uploaded {local_path} to {dropbox_path}")

    def download_file(self, dropbox_path, local_path):
        """Download a single file from Dropbox to local directory."""
        try:
            self.dbx.files_download_to_file(local_path, dropbox_path)
            print(f"Downloaded {dropbox_path} to {local_path}")
        except Exception as e:
            print(f"Error downloading {dropbox_path}: {e}")

    def download_folder(self, dropbox_folder_path, local_folder_path):
        """Download the contents of a Dropbox folder recursively."""
        try:
            for entry in self.dbx.files_list_folder(dropbox_folder_path).entries:
                local_path = os.path.join(local_folder_path, entry.name)
                if isinstance(entry, dropbox.files.FileMetadata):
                    self.download_file(entry.path_lower, local_path)
                elif isinstance(entry, dropbox.files.FolderMetadata):
                    if not os.path.exists(local_path):
                        os.makedirs(local_path)
                    self.download_folder(entry.path_lower, local_path)
        except Exception as e:
            print(f"Error accessing {dropbox_folder_path}: {e}")

