import os

def list_files(dirName):
    """return a list of image filenames with path from a directory. Works with nested or unnested files.
    
    args:
        dirName -- the directory from which to list images
    
    """
    
    listOfFiles = list()
    for (dirpath, dirnames, filenames) in os.walk(dirName):
        listOfFiles += [os.path.join(dirpath, file) for file in filenames]
            
    return listOfFiles 

    

