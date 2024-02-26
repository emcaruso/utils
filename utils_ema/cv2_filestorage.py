# import cv2
# # import json


# def write_yml(path, dictionary):
#     fs = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
#     write_yml_aux(fs, "", dictionary)
#     fs.release()

# def write_yml_aux(fs, base_key, dictionary):

#     # base_key = base_key.split("/")[-1]

#     for key, value in dictionary.items():
#         print(base_key)
#         full_key = base_key+"{"+key+"}"
#         print(value)
#         if isinstance(value, dict):
#             write_yml_aux(fs, full_key, value)
#         elif isinstance(value, list):
#             # Handle list of dictionaries (assuming all elements are dictionaries for simplicity)
#             for i, item in enumerate(value):
#                 item_key = full_key+"{"+i+"}"
#                 write_yml_aux(fs, item_key, item)
#         else:
#             print(full_key)
#             fs.write(full_key, 1)

# # def write_yml(path, dictionary):
#     # # Open a FileStorage object in write mode
#     # fs = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)

#     # # Convert the nested dictionary to a JSON string
#     # data_json = json.dumps(dictionary)

#     # # Write the JSON string to the file under a compliant key name
#     # fs.write("data", data_json)

#     # # Release the file storage
#     # fs.release()

