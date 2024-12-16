import os


class CustomOS:
    def __getattr__(self, name):
        return getattr(os, name)
    
    def listdir(self, dir_pth, sort=False):
        dir_items = os.listdir(dir_pth)
        return sorted(dir_items, key=lambda x: int(x)) if sort else dir_items
    
custom_os = CustomOS()