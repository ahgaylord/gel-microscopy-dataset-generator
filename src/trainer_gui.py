""" Python GUI used for generating datasets from a video, image, or set of files.

@author Amory Gaylord
"""

# imports
import os
from tkinter import * # wildcard: ignore
from tkinter import ttk
from tkinter import filedialog
from tkVideoPlayer import TkinterVideo as tktv
import labeler
import cv2
import numpy as np
from PIL import ImageTk, Image
import matplotlib.pyplot as plt
from roboflow import Roboflow
from enum import Enum

class ProgramState(Enum):
    """ ProgramState enumeration defines unique integers for different pre-defined program states.
    """    
    
    WELCOME = 0
    SELECT_SOURCE = 1
    SELECT_NAME = 3
    SELECT_SAVE = 10
    SELECT_PARAMETERS = 4
    PROCESS_DATASET = 5
    PREVIEW_DATASET = 6
    UPLOAD_DATASET = 7
    UPLOAD_SUCCESS = 8
    UPLOAD_FAILURE = 9

class ButtonPress(Enum):
    """ ButtonPress enumeration defines unique integers for different pre-defined user commands.
    """  

    EXIT = 0
    GENERATE = 1
    UPLOAD = 2
    CONTINUE = 3
    CHANGE_SRC = 4
    CHANGE_NAME = 5
    CHANGE_SAVE = 6
    PREVIEW = 7
    START_OVER = 8
    RETRY = 9
    CHANGE_COLOR = 10


class Program:
    """ Handles backend program flow for dataset GUI.
    
    Attributes:
    -----------
    state : int
        The current program state. Reference class ProgramState.
    filenames : list
        The list of source files associated with this Program.
    upload_only : bool
        True if the program is dealing with Roboflow only, False if it is dealing with generation
        and upload tasks.
    dataset_name : str
        The name of the dataset used for RF and directory purposes.
    save_dir : str
        Pathname of the directory to save previews to.
    lower_bound : ndarray
        Numpy array containing the lower threshold for segmentation in BGR format.
    upper_bound : ndarray
        Numpy array containing the upper threshold for segmentation in BGR format.
    eps : float
        Epsilon value used for cv2 polygon outline approximation. Defined as the maximum distance
        between 2 polygon outline vertices.
    contrast : int
        Amount to increase contrast by during preprocessing.
    brighntess : int
        Amount to increase brightness during preprocessing.
    gaussian : int
        Amount to blur during preprocessing.
    segments : int
        The number of blocks to segment an image into during preprocessing.
    mcs : int
        The minimum number of pixels for a cluster to be included as a cluster.
    upload_error : str
        Error message during the upload task, if an error occurred.

    """

    VALID_CMDS = [ButtonPress.EXIT, ButtonPress.GENERATE, ButtonPress.UPLOAD, ButtonPress.CONTINUE, ButtonPress.CHANGE_SRC, ButtonPress.CHANGE_NAME, ButtonPress.CHANGE_SAVE, ButtonPress.PREVIEW, ButtonPress.START_OVER, ButtonPress.RETRY, ButtonPress.CHANGE_COLOR]

    def __init__(self):
        """ Initializes attributes to correct start values.

        Returns
        -------
        None.

        """        

        self.state = ProgramState.WELCOME
        self.filenames = []
        self.upload_only = False
        self.dataset_name = ""
        self.save_dir = ""
        self.lower_bound = np.array([0, 0, 0])
        self.upper_bound = np.array([255, 255, 255])
        self.eps = labeler.EPS
        self.contrast = labeler.CONTRAST
        self.brightness = labeler.BRIGHTNESS
        self.gaussian = labeler.GAUSSIAN
        self.segments = labeler.SEGMENTS
        self.mcs = labeler.MIN_CLUSTER_SIZE
        self.upload_error = ""        

    def handle(self, cmd, args=[]):
        """ Handles commands and additional arguments (if any) based on the current program state.

        Parameters
        ----------
        cmd : int
            The user command passed from the GUI. Reference class ButtonPress.
        args : list, optional
            List of additional arguments (e.g. parameters, selections, etc.). The default value is 
            [].

        Returns
        -------
        Any.

        Raises
        ------
        ValueError
            Incorrect command given current state or incorrect number/type of args for the command/
            state.

        """

        # exit requires no special handling
        if cmd == ButtonPress.EXIT:
            exit()

        # readability
        s = self.state

        # check parameter cmd
        if cmd not in self.VALID_CMDS:
            raise ValueError(str(cmd) + " is not associated with a command.")

        # invalid command error
        err_str = "Command <" + cmd.name + "> is not a valid command for state <" + s.name + ">."
        
        # default program start state
        if s == ProgramState.WELCOME:
            self.welcome(cmd, err_str)
    
        # selecting source files
        elif s == ProgramState.SELECT_SOURCE:
            self.select_source(cmd, err_str, args)
        
        # currently selecting a save
        elif s == ProgramState.SELECT_SAVE:
            self.select_save(cmd, err_str, args)
        
        # entering a name for the dataset
        elif s == ProgramState.SELECT_NAME:
            self.select_name(cmd, err_str, args)

        # parameter selection
        elif s == ProgramState.SELECT_PARAMETERS:
            return self.select_parameters(cmd, err_str, args)
        
        # processing dataset files
        elif s == ProgramState.PROCESS_DATASET:
            self.process_dataset(cmd, err_str)
        
        # previewing annotated files
        elif s == ProgramState.PREVIEW_DATASET:
            self.preview_dataset(cmd, err_str)
        
        # uploading dataset to Roboflow
        elif s == ProgramState.UPLOAD_DATASET:
            self.upload_dataset(cmd, err_str, args)
        
        # upload failed
        elif s == ProgramState.UPLOAD_FAILURE:
            self.upload_failure(cmd, err_str)
        
        # successful upload
        elif s == ProgramState.UPLOAD_SUCCESS:
            self.upload_success(cmd, err_str)

    # handle subroutines

    def welcome(self, cmd, err_str):
        """ Handles input for state WELCOME.

        Parameters
        ----------
        cmd : int
            The user command passed from the GUI. Reference class ButtonPress.
        err_str : str
            Invalid command message for this state and command.

        Returns
        -------
        None.

        Raises
        ------
        ValueError
            Invalid command or args for this state.

        """ 

        # valid_commands = [ButtonPress.GENERATE, ButtonPress.UPLOAD, ButtonPress.EXIT]

        # generating a dataset
        if cmd == ButtonPress.GENERATE:
            self.state = ProgramState.SELECT_SOURCE
            self.upload_only = False

        # only uploading a pre-existing dataset            
        elif cmd == ButtonPress.UPLOAD:
            self.state = ProgramState.SELECT_SOURCE
            self.upload_only = True
        
        # invalid command for WELCOME state
        else:
            raise ValueError(err_str)
        
    def select_source(self, cmd, err_str, args):
        """ Handles input for state SELECT_SOURCE.

        Parameters
        ----------
        cmd : int
            The user command passed from the GUI. Reference class ButtonPress.
        err_str : str
            Invalid command message for this state and command.
        args : list
            List of additional arguments (e.g. parameters, selections, etc.).

        Returns
        -------
        None.

        Raises
        ------
        ValueError
            Invalid command or args for this state.

        """

        # valid_commands = [ButtonPress.CHANGE_SRC, ButtonPress.EXIT]
            
        # assign source files
        if cmd == ButtonPress.CHANGE_SRC:
            # handles all possible args (single file, multi file, folders)
            self.filenames = self.get_file_list(args[0])

            # update state based on whether previous selection was for generation or upload only
            if self.upload_only:
                self.state = ProgramState.SELECT_NAME
            else:
                self.state = ProgramState.SELECT_SAVE
            
        # invalid command for SELECT_SOURCE state
        else:
            raise ValueError(err_str)
    
    def select_save(self, cmd, err_str, args):
        """ Handles input for state SELECT_SAVE.

        Parameters
        ----------
        cmd : int
            The user command passed from the GUI. Reference class ButtonPress.
        err_str : str
            Invalid command message for this state and command.
        args : list
            List of additional arguments (e.g. parameters, selections, etc.).

        Returns
        -------
        None.

        Raises
        ------
        ValueError
            Invalid command or args for this state.

        """

        # valid_commands = [ButtonPress.CHANGE_SAVE, ButtonPress.EXIT]

        # assign new save
        if cmd == ButtonPress.CHANGE_SAVE:
            # handle args
            self.change_save_dir(args[0])
            # update state
            self.state = ProgramState.SELECT_NAME

        # invalid command for SELECT_SAVE state
        else:
            raise ValueError(err_str)
    
    def select_name(self, cmd, err_str, args):
        """ Handles input for state SELECT_NAME.

        Parameters
        ----------
        cmd : int
            The user command passed from the GUI. Reference class ButtonPress.
        err_str : str
            Invalid command message for this state and command.
        args : list
            List of additional arguments (e.g. parameters, selections, etc.).

        Returns
        -------
        None.

        Raises
        ------
        ValueError
            Invalid command or args for this state.

        """

        # valid_commands = [ButtonPress.CHANGE_NAME, ButtonPress.CHANGE_SRC, ButtonPress.CHANGE_SAVE, ButtonPress.EXIT]

        # default continue with entry command
        if cmd == ButtonPress.CHANGE_NAME:
            # handle args
            self.change_name(args[0])

            # update state based on program upload status
            if self.upload_only:
                self.state = ProgramState.UPLOAD_DATASET
            else:
                self.state = ProgramState.SELECT_PARAMETERS

        # step backward, update source files
        elif cmd == ButtonPress.CHANGE_SRC:
            self.state = ProgramState.SELECT_SOURCE

        # step backward, update save directory (if upload status is False)
        elif cmd == ButtonPress.CHANGE_SAVE and not self.upload_only:
            self.state = ProgramState.SELECT_SAVE

        # invalid command for SELECT_NAME state
        else:
            raise ValueError(err_str)

    def select_parameters(self, cmd, err_str, args):
        """ Handles input for state SELECT_PARAMETERS.

        Parameters
        ----------
        cmd : int
            The user command passed from the GUI. Reference class ButtonPress.
        err_str : str
            Invalid command message for this state and command.
        args : list
            List of additional arguments (e.g. parameters, selections, etc.).

        Returns
        -------
        None.

        Raises
        ------
        ValueError
            Invalid command or args for this state.

        """

        # valid_commands = [ButtonPress.PREVIEW, ButtonPress.CONTINUE, ButtonPress.CHANGE_NAME, ButtonPress.CHANGE_SAVE]

        # update dataset name
        if cmd == ButtonPress.CHANGE_NAME:
            self.change_name(args[0])

        # step backward, change save directory
        elif cmd == ButtonPress.CHANGE_SAVE:
            self.state = ProgramState.SELECT_SAVE

        # step backward, change source files
        elif cmd == ButtonPress.CHANGE_SRC:
            self.src = ProgramState.SELECT_SOURCE

        # update preview image
        elif cmd  == ButtonPress.PREVIEW:
            # validate number of passed args
            if len(args) != 7:
                raise ValueError("Incorrect args value. Args must have 7 items (image, eps, contrast, brightness, gaussian, segments, and mcs).")
            
            # get values
            img = args[0]
            self.eps = args[1]
            self.contrast = args[2]
            self.brightness = args[3]
            self.gaussian = args[4]
            self.segments = args[5]
            self.mcs = args[6]

            # update preview image
            return self.preview_frame(img)
        
        # change threshold color
        elif cmd == ButtonPress.CHANGE_COLOR:
            # validate number of passed args
            if len(args) != 2:
                raise ValueError("Incorrect args value. Args must have 2 items (is_lower and threshold).")

            # get values
            is_lower = args[0]
            threshold = args[1]

            # update threshold
            self.assign_bound(threshold, is_lower)
        
        # accept parameters and continue to processing step
        elif cmd == ButtonPress.CONTINUE:
            # validate number of passed args
            if len(args) != 6:
                raise ValueError("Incorrect args value. Args must have 6 items (eps, contrast, brightness, gaussian, segments, and mcs).")

            # get values
            self.eps = args[0]
            self.contrast = args[1]
            self.brightness = args[2]
            self.gaussian = args[3]
            self.segments = args[4]
            self.mcs = args[5]

            # update state
            self.state = ProgramState.PROCESS_DATASET

        # step backwards, start from Welcome
        elif cmd == ButtonPress.START_OVER:
            self.state = ProgramState.WELCOME

        # invalid command for SELECT_PARAMETERS state
        else:
            raise ValueError(err_str)
    
    def process_dataset(self, cmd, err_str):
        # continue to preview state
        if cmd == ButtonPress.CONTINUE:
            self.state = ProgramState.PREVIEW_DATASET
        
        # invalid command for PROCESS_DATASET state
        else:
            raise ValueError(err_str)
    
    def preview_dataset(self, cmd, err_str):
        """ Handles input for state PREVIEW_DATASET.

        Parameters
        ----------
        cmd : int
            The user command passed from the GUI. Reference class ButtonPress.
        err_str : str
            Invalid command message for this state and command.

        Returns
        -------
        None.

        Raises
        ------
        ValueError
            Invalid command or args for this state.

        """

        # step back, start from Welcome state
        if cmd == ButtonPress.START_OVER:
            self.state = ProgramState.WELCOME
        
        # continue to upload state
        elif cmd == ButtonPress.UPLOAD:
            self.state = ProgramState.UPLOAD_DATASET
        
        # invalid command for PREVIEW_DATASET state
        else:
            raise ValueError(err_str)
    
    def upload_dataset(self, cmd, err_str, args):
        """ Handles input for state UPLOAD_DATASET.

        Parameters
        ----------
        cmd : int
            The user command passed from the GUI. Reference class ButtonPress.
        err_str : str
            Invalid command message for this state and command.
        args : list
            List of additional arguments (e.g. parameters, selections, etc.).

        Returns
        -------
        None.

        Raises
        ------
        ValueError
            Invalid command or args for this state.

        """

        # confirm upload to Roboflow
        if cmd == ButtonPress.UPLOAD:
            # validate number of passed args
            if len(args) != 5:
                raise ValueError("Incorrect args value. Args must have 5 items (api key, imgs, annotations, project_id, and batch_name).")
            
            # get values
            api_key = args[0]
            imgs = args[1]
            annotations = args[2]
            project_id = args[3]
            batch_name = args[4]
            
            # handle upload 
            self.upload_files(api_key, imgs, annotations, batch_name, project_id)
            # upload_files automatically updates status
        
        # start from beginning
        elif cmd == ButtonPress.START_OVER:
            self.state = ProgramState.WELCOME
        
        # invalid command for UPLOADING state
        else:
            raise ValueError(err_str)
    
    def upload_failure(self, cmd, err_str):
        """ Handles input for state UPLOAD_FAILURE.

        Parameters
        ----------
        cmd : int
            The user command passed from the GUI. Reference class ButtonPress.
        err_str : str
            Invalid command message for this state and command.

        Returns
        -------
        None.

        Raises
        ------
        ValueError
            Invalid command or args for this state.

        """

        # retry upload, returns to Uploading state
        if cmd == ButtonPress.RETRY:
            self.state = ProgramState.UPLOAD_DATASET

        # start from beginning
        elif cmd == ButtonPress.START_OVER:
            self.state == ProgramState.WELCOME

        # not a valid command for UPLOAD_FAILURE state
        else:
            raise ValueError(err_str)
    
    def upload_success(self, cmd, err_str):
        """ Handles input for state UPLOAD_SUCCESS.

        Parameters
        ----------
        cmd : int
            The user command passed from the GUI. Reference class ButtonPress.
        err_str : str
            Invalid command message for this state and command.

        Returns
        -------
        None.

        Raises
        ------
        ValueError
            Invalid command or args for this state.

        """

        # start from beginning
        if cmd == ButtonPress.START_OVER:
            self.state = ProgramState.WELCOME

        # not a valid command for UPLOAD_SUCCESS state
        else:
            raise ValueError(err_str)

    # start validation and backend processing functions

    def preview_frame(self, frame):
        """ Gets a frame annotated with the program's current processing parameters and returns it.

        Parameters
        ----------
        frame : Mat
            The image to get a preview of.

        Returns
        -------
        preview : Mat
            A preview of the frame with the program's processing parameters.

        """    
        
        preview = labeler.get_clusters(frame, self.lower_bound, self.upper_bound, self.eps, self.mcs, self.contrast, self.brightness, self.gaussian, self.segments, False, labeler.INTERMEDIATE_DESTINATION, labeler.DRAW_DESTINATION, False, True, return_frame=True)
        return preview

    def assign_bound(self, bound, is_lower):
        """ Assigns the input bound to the program's lower or upper threshold attributes.

        Parameters
        ----------
        bound : list OR ndarray
            The bound to update the given attribute with.
        is_lower : bool
            True if updated the lower threshold, False if updating the upper threshold.

        Raises
        ------
        TypeError
            Invalid bound type.
        ValueError
            Invalid bound shape (must be 3x1).
        ValueError
            Value of bound exceeded 255 or is less than 0.
        """

        # initialize arr   
        arr = np.array([0, 0, 0], np.uint8)

        # check type and assign bound to arr
        if type(bound) is list:
            arr = np.array(bound, np.uint8)
        elif type(bound) is np.array:
            arr = bound
        else:
            raise TypeError("A bound must be a list or a Numpy array. The passed argument was type " + str(type(bound)) + ".")

        # check shape
        if arr.shape[0] != 3 and arr.shape[1] != 1:
            raise ValueError("Incorrect dimensions for a bound. Bounds should have shape 3x1. The passed argument had shape " + str(arr.shape[0]) + "x" + str(arr.shape[1]) + ".")
        
        # check values
        for val in arr:
            if val < 0 or val > 255:
                raise ValueError("Values in a threshold must be between 0 and 255, inclusive. Value " + str(val) + " is out of range.")
        
        # checks passed: assign bound
        if is_lower:
            self.lower_bound = arr
        else:
            self.upper_bound = arr

    def change_name(self, name):
        """ Changes the program's dataset name.

        Parameters
        ----------
        name : str
            Name of the dataset.

        Raises
        ------
        ValueError
            A name cannot be an empty string.
        TypeError
            A name must be a string.

        """
        
        # check type and assign
        if type(name) is str and len(name) > 0:
            self.dataset_name = name
        elif len(name) <= 0:
            raise ValueError("A name cannot be an empty string.")
        else:
            raise TypeError("Names must be strings. The passed parameter had type " + str(type(name)) + ".")

    def change_save_dir(self, d):
        """ Update the program's save directory.

        Parameters
        ----------
        d : str
            Path to the directory to set as the save directory.

        Raises
        ------
        TypeError
            d must be a string. No other types are accepted.
        """    

        # check type
        if type(d) is str:
            if not os.path.exists(d): 
                os.mkdir(d)
            
            self.save_dir = d
        else:
            raise TypeError("A directory pathname must be a string. Type " + str(type(d)) + " is not a valid directory.")

    def remove_files(self, files):
        """ Removes every file in the list of files.

        Parameters
        ----------
        files : list
            List of pathnames to remove. They must be sequential (e.g. if a folder is listed before 
            one of its subfolders, errors will occur).

        """ 

        for f in files:
            os.remove(f)

    def upload_files(self, api_key, imgs, annotations, name, project_id):
        """ Uploads files to Roboflow.

        Parameters
        ----------
        api_key : str
            The private api key associated with the target workspace.
        imgs : str
            Pathname for the directory containing images to upload.
        annotations : str
            Pathname to the file containing the annotations for the images to upload.
        name : str
            Batch name for upload.
        project_id : str
            The target project's ID, either its name or its name-[IDENTIFIER] (in cases where there 
            may be multiple projects with the same name).

        """        

        # Initialize the Roboflow object with your API key
        rf = Roboflow(api_key=api_key)
        
        # Specify the project for upload
        project = rf.workspace(rf.current_workspace).project(project_id)

        # get a list of images
        f_list = os.listdir(imgs)

        try:
            # for every file
            for f in f_list:
                # pathname is the folder plus the filename
                p = imgs + "/" + f
                # upload image and annotations
                project.upload(image_path=p, annotation_path=annotations, num_retry_uploads=3, batch_name=name, tags=[self.dataset_name])
    
        # catch exception as e
        except Exception as e:
            # get upload error message
            message = str(e)
            # update error message attribute
            self.upload_error = message
            # update state
            self.state = ProgramState.UPLOAD_FAILURE
            # exit out of loop
            return
        
        # no errors encountered, update state to Success
        self.state = ProgramState.UPLOAD_SUCCESS

    def handle_directory(self, d, list_files):
        """ Adds all files from a directory to the list_files.

        Parameters
        ----------
        d : str
            Pathname for the directory to get files from.
        list_files : list
            List of files to add files to.

        Returns
        -------
        list_files : list
            The passed parameter updated with the new files.

        Raises
        ------
        TypeError
            If the argument passed as d is not a str.
        OSError
            If the requested directory does not exist.
        TypeError
            If the argument passed as list_files is not a list.
        OSError
            If a file in the directory does not exist.

        """
        
        # type checks
        if type(d) is not str:
            raise TypeError("Input directory must be of type str. Type " + str(type(d)) + "is not an acceptable format.")
        if not os.path.exists(d):
            raise OSError("Input directory <" + d + "> does not exist.")
        if type(list_files) is not list:
            raise TypeError("list_files must be of type list. Type " + str(type(list_files)) + "is not an acceptable format.")

        # for each file...
        for f in os.listdir(d):
            # check path
            if not os.path.exists(f):
                raise OSError("Not a valid path")
            
            # append if file
            elif os.path.isfile(f):
                list_files.append(f)

            # recursive call if directory
            elif os.path.isdir(f):
                list_files = self.handle_directory(f, list_files)

        return list_files

    def get_file_list(self, files):
        """ Gets a list of files from an input of a list or a pathname.

        Parameters
        ----------
        files : str OR list
            Either a list of files or a pathname to a single file or directory.

        Returns
        -------
        list_files : list
            A list of files.

        Raises
        ------
        TypeError
            If the input is in an invalid format.
        OSError
            If the requested file does not exist.

        """ 

        # initialize list
        list_files = []

        if type(files) is list:
            # iterate through the list
            for f in files:
                # check type
                if type(f) is not str:
                    raise TypeError("List items must be strings. Type " + str(type(f)) + " is not an acceptable format.")
                
                # check path exists
                if not os.path.exists(f):
                    raise OSError("Path <" + f + "> does not exist.")
                
                # if file...
                elif os.path.isfile(f):
                    list_files.append(f)
                
                # if directory...
                elif os.path.isdir(f):
                    list_files = self.handle_directory(f, list_files)

        elif type(files) is str:
            # check path exists
            if not os.path.exists(files):
                raise OSError("Path <" + files + "> does not exist.")
            
            # if directory...
            if os.path.isdir(files):
                list_files = self.handle_directory(files, list_files)
            
            # if file...
            elif os.path.isfile(files):
                list_files = list_files.append(files)

        else:
            raise TypeError("Input must be a list or a pathname. Type " + str(type(files)) + " is not an acceptable format.")
        
        return list_files


# start GUI classes

class TrainingDataGenerator:
    """ Parent GUI class.

    Attributes
    ----------
    root : Tk
        Top level Tkinter widget (e.g. the program).
    program : Program
        Handles program state and back end management.
    active_frame : ttk.Frame
        Tkinter frame (window) that is currently active.
    mainframe : ttk.Frame
        Tkinter frame (window) that is associated with the Welcome window.

    """

    def __init__(self, root):
        """ Initializes the TrainingDataGenerator GUI.

        Parameters
        ----------
        root : Tk
            Top-level Tkinter widget.

        Returns
        -------
        None.

        """        

        # set title
        root.title("Training Data Generator")

        # set up window
        mainframe = ttk.Frame(root, padding="3 3 12 12")
        mainframe.pack(expand=True)

        # set attributes
        self.root = root
        self.program = Program()
        self.active_frame = mainframe

        # configure window
        self.root.maxsize(root.winfo_screenwidth(), root.winfo_screenheight())

        # define welcome message
        welcome_msg = "Welcome to the Hsiao Lab dataset generation program!\nTo generate a dataset from a video or image, click GENERATE. To upload an existing dataset, click UPLOAD."
        ttk.Label(mainframe, text=welcome_msg).pack(side="top", expand=True)

        # create control panel
        ctrl_panel = Canvas(mainframe)
        ctrl_panel.pack(side="bottom", expand=True)

        # create control panel buttons
        ttk.Button(ctrl_panel, text="Generate", command=self.generate).pack(side="left", expand=True)
        ttk.Button(ctrl_panel, text="Exit", command=self.finish).pack(side="right", expand=True)
        ttk.Button(ctrl_panel, text="Upload", command=self.upload).pack(side="right", expand=True)

        # key binds
        root.bind("<Configure>", self.conf)
        root.bind("<Return>", self.generate)

        # configure window
        mainframe.config(height=root.winfo_screenheight(), width=root.winfo_screenwidth())

        # set attribute
        self.mainframe = mainframe
        
    def cont(self):
        """ Changes the active frame based on the current program state.

        Returns
        -------
        None.

        """        
        
        # destroy the active frame
        if self.active_frame.winfo_exists():
            self.active_frame.destroy()

        # set the active frame based on the current state
        if self.program.state == ProgramState.WELCOME:
            self.active_frame = TrainingDataGenerator(self.root).mainframe

        elif self.program.state == ProgramState.SELECT_SOURCE:
            self.active_frame = SelectSource(self, self.root).mainframe

        elif self.program.state == ProgramState.SELECT_SAVE:
            self.get_save_folder()
            self.cont()

        elif self.program.state == ProgramState.SELECT_NAME:
            self.active_frame = NameSelector(self, self.root).mainframe

        elif self.program.state == ProgramState.SELECT_PARAMETERS:
            self.active_frame = ParamsSelector(self, self.root).mainframe

        elif self.program.state == ProgramState.PROCESS_DATASET:
            self.active_frame = Processing(self, self.root).mainframe

        elif self.program.state == ProgramState.PREVIEW_DATASET:
            self.active_frame = Previewing(self, self.root).mainframe

        elif self.program.state == ProgramState.UPLOAD_DATASET:
            self.active_frame = Uploading(self, self.root).mainframe

        elif self.program.state == ProgramState.UPLOAD_FAILURE:
            self.active_frame = UploadFailure(self, self.root).mainframe

        elif self.program.state == ProgramState.UPLOAD_SUCCESS:
            self.active_frame = UploadSuccess(self, self.root).mainframe

    def bound_to_str(self, bound):
        """ Converts a 3x1 Numpy array to a string.

        Parameters
        ----------
        bound : list or list-like (can be indexed)
            3x1 list to convert to a string.

        Returns
        -------
        str
            String version of the input bound.

        """ 

        return str(bound[2]) + ", " + str(bound[1]) + ", " + str(bound[0])
    
    def get_save_folder(self):
        """ Gets a save folder and passes the path to the program to be handled.

        Returns
        -------
        None.

        """        
        
        # prompt user with file dialog
        directory = filedialog.askdirectory(mustexist=False, title="Select a save directory")
        # handle input
        self.program.handle(ButtonPress.CHANGE_SAVE, args=[directory])
        # continue
        self.cont()

    def finish(self):
        """ Exits the program.

        Returns
        -------
        None.

        """
        
        exit()

    def upload(self, *args):
        """ Handles an upload button press.

        Returns
        -------
        None.

        """

        self.program.handle(ButtonPress.UPLOAD)
        self.cont()
    
    def generate(self, *args):
        """ Handles a generate button press.

        Returns
        -------
        None.

        """

        self.program.handle(ButtonPress.GENERATE)
        self.cont()

    def conf(self, *kwargs):
        """ Configures the root and the active frame.

        Returns
        -------
        None.

        """

        # win = self.
        # self.root.update
        if self.active_frame.winfo_exists():
            self.active_frame.config(height=int(self.root.winfo_screenheight() * 0.75), width=int(self.root.winfo_screenwidth() * 0.75))

        # self.root.attributes('-fullscreen',True)

        # self.root.eval('tk::PlaceWindow %s center' % self.root.winfo_pathname(self.root.winfo_id()))

        # self.active_frame.place
        # screen_width = self.root.winfo_screenwidth()
        # screen_height = self.root.winfo_screenheight()

        # x_cordinate = int((screen_width/2) - (self.root.winfo_screenwidth()/2))
        # y_cordinate = int((screen_height/2) - (self.root.winfo_screenheight()/2))

        # win.geometry("{}x{}+{}+{}".format(window_width, window_height, x_cordinate, y_cordinate))


class SelectSource:
    """ Window for selecting source files.

    Attributes
    ----------
    parent : TrainingDataGenerator
        The parent window.
    root : Tk
        The top-level Tkinter widget associated with this GUI.
    mainframe : ttk.Frame
        The window associated with this class.

    """ 
    
    def __init__(self, parent, root):
        """ Initializes the Source Selection window.

        Parameters
        ----------
        parent : TrainingDataGenerator
            The parent window.
        root : Tk
            The top-level Tkinter widget associated with this GUI.

        """        
    
        # set up attributes
        self.parent = parent
        self.root = root

        # configure window
        mainframe = ttk.Frame(self.root, padding="3 3 12 12")
        mainframe.pack(expand=True)

        # set up instructions panel
        instructions = "Select source file(s) and a save directory."
        ttk.Label(mainframe, text=instructions).pack(side="top", expand=True)

        # set up control panel
        ctrl_panel = Canvas(mainframe)
        ctrl_panel.pack(side="bottom", expand=True)

        # create buttons
        ttk.Button(ctrl_panel, text="Select multiple source files", command=self.select_multiple).pack(side="left")
        ttk.Button(ctrl_panel, text="Select source file", command=self.select_single).pack(side="left", expand=True)
        ttk.Button(ctrl_panel, text="Exit", command=self.parent.finish).pack(side="right", expand=True)
        ttk.Button(ctrl_panel,text="Select source folder", command=self.select_folder).pack(side="right", expand=True)

        # set attribute
        self.mainframe = mainframe

        # key binds
        root.bind("<Return>", self.select_single)

    def select(self, f):
        """ Passes f to the program to be handled.

        Parameters
        ----------
        f : any
            The file(s) to be handled.

        Returns
        -------
        None.
        
        """

        self.parent.program.handle(ButtonPress.CHANGE_SRC, args=[f])
        self.parent.cont()
    
    def select_single(self, *kwargs):
        """ Opens a file dialog for selecting a single file.

        Returns
        -------
        None.

        """

        # open file dialog
        filename = filedialog.askopenfilename()

        # if invalid selection, wait for user to re-enter.
        try:
            self.select([filename])
        except:
            pass
    
    def select_multiple(self):
        """ Opens a file dialog for selecting multiple files.
        
        Returns
        -------
        None.

        """

        # open file dialog        
        files = filedialog.askopenfilenames()

        # if invalid selection, wait for user to re-enter
        try:
            self.select(files)
        except:
            pass
    
    def select_folder(self):
        """ Opens a file dialog for selecting a directory.

        Returns
        -------
        None.

        """

        # open file dialog
        repo = filedialog.askdirectory()
        
        # if invalid selection, wait for user to re-enter
        try:
            self.select([repo])
        except:
            pass


class NameSelector:
    """ Window for entering the dataset name.

    Attributes
    ----------
    parent : TrainingDataGenerator
        The parent window.
    root : Tk
        The top-level Tkinter widget associated with this GUI.
    name_var : StringVar
        String variable containing the user's input dataset name and tied to the entry widget.
    mainframe : ttk.Frame
        The window associated with this class.

    """ 

    def __init__(self, parent, root):
        """ Initializes the dataset name window.

        Parameters
        ----------
        parent : TrainingDataGenerator
            The parent program handling the GUI.
        root : Tk
            The top-level Tkinter widget associated with this GUI.

        Returns
        -------
        None.

        """ 

        # set attributes        
        self.parent = parent
        self.root = root

        # set up window
        mainframe = ttk.Frame(self.root, padding="3 3 12 12")
        mainframe.pack(expand=True)

        # create var
        self.name_var = StringVar(mainframe, self.parent.program.dataset_name)

        # set up panels
        name_panel = Canvas(mainframe)
        name_panel.pack(side="top", expand=True)
        ctrl_panel = Canvas(mainframe)
        ctrl_panel.pack(side="bottom", expand=True)

        # create label and entry
        ttk.Label(name_panel, text="dataset name").pack(side="left", expand=True)
        ttk.Entry(name_panel, textvariable=self.name_var).pack(side="right", expand=True)

        # create buttons
        ttk.Button(ctrl_panel, text="continue", command=self.cont).pack(side="left", expand=True)
        ttk.Button(ctrl_panel, text="change source directory", command=self.change_src).pack(side="left", expand=True)
        ttk.Button(ctrl_panel, text="change save directory", command=self.change_save).pack(side="left", expand=True)

        # key binds
        root.bind("<Return>", self.cont)

        # set attribute
        self.mainframe = mainframe

    def cont(self, *kwargs):
        """ Handles progressing to the next state in the GUI. Gets the var and passes it to the
        program.

        Returns
        -------
        None.

        """

        # if handle raises an exception, wait for new input
        try:
            self.parent.program.handle(ButtonPress.CHANGE_NAME, args=[self.name_var.get()])
            self.parent.cont()
        except:
            pass

    def change_src(self):
        """ Handles the change_src button press and passes it to the program.
        
        Returns
        -------
        None.

        """

        self.parent.program.handle(ButtonPress.CHANGE_SRC)
        self.parent.cont()
    
    def change_save(self):
        """ Handles the change_save button press and passes it to the program.

        Returns
        -------
        None.

        """        
        self.parent.program.handle(ButtonPress.CHANGE_SAVE)
        self.parent.cont()


class ParamsSelector:
    """ Window for entering the preprocessing parameters.

    Attributes
    ----------
    parent : TrainingDataGenerator
        The parent window.
    root : Tk
        The top-level Tkinter widget associated with this GUI.
    program : Program
        The back-end state machine handling the commands passed from the GUI.
    img : Mat
        Original image used for color selection with Matplotlib.
    original_frame : PhotoImage
        Original image used for internal storage.
    brightness_var : IntVar
        Variable containing the user's selected brightness value.
    gaussian_var : IntVar
        Variable containing the user's selected gaussian value.
    contrast_var : DoubleVar
        Variable containing the user's selected contrast value.
    epsilon_var : DoubleVar
        Variable containing the user's selected epsilon value.
    segments_var : IntVar
        Variable containing the user's selected number of segments.
    mcs_var : IntVar
        Variable containing the user's selected min cluster size.
    lower_var : StringVar
        Variable containing the lower color threshold in string format.
    upper_var : StringVar
        Variable containing the upper color threshold in string format.
    preview_frame : PhotoImage
        The preview frame used for internal storage.
    name : StringVar
        Variable containing the user's selected dataset name.
    preview_img_display : ttk.Label
        The widget containing the preview image when displaying.
    preview_panel : Canvas
        The panel containing the preview image label and other widgets.
    lower_r : IntVar
        Variable containing the user's selected lower red value.
    lower_g : IntVar
        Variable containing the user's selected lower green value.
    lower_b : IntVar
        Variable containing the user's selected lower blue value.
    upper_r : IntVar
        Variable containing the user's selected upper red value.
    upper_g : IntVar
        Variable containing the user's selected upper green value.
    upper_b : IntVar
        Variable containing the user's selected upper blue value.
    mainframe : ttk.Frame
        The window associated with this class.
    
    """ 
    
    def __init__(self, parent, root):
        """ Initializes the parameter selection window.

        Parameters
        ----------
        parent : TrainingDataGenerator
            The parent window.
        root : Tk
            The top-level Tkinter widget associated with this GUI.

        Returns
        -------
        None.

        """        
        
        # set attributes
        self.parent = parent
        self.root = root
        self.program = parent.program

        # set up window
        mainframe = ttk.Frame(self.root, padding="3 3 12 12")
        mainframe.pack(fill="both", expand=True)
        self.mainframe = mainframe

        # set up original image variables
        self.img = self.get_preview_img_original()
        rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB) # type: ignore
        shape = self.get_image_size(rgb)
        pil_img = Image.fromarray(rgb).resize(shape) # type: ignore
        self.original_frame = ImageTk.PhotoImage(pil_img)

        # set up preprocessing parameter variables
        self.brightness_var = IntVar(mainframe, self.program.brightness)
        self.gaussian_var = IntVar(mainframe, self.program.gaussian)
        self.contrast_var = DoubleVar(mainframe, self.program.contrast)
        self.epsilon_var = DoubleVar(mainframe, self.program.eps)
        self.segments_var = IntVar(mainframe, self.program.segments)
        self.mcs_var = IntVar(mainframe, self.program.mcs)
        self.lower_var = StringVar(mainframe, self.parent.bound_to_str(self.program.lower_bound))
        self.upper_var = StringVar(mainframe, self.parent.bound_to_str(self.program.upper_bound))

        args = [self.img, self.epsilon_var.get(), self.contrast_var.get(), self.brightness_var.get(), self.gaussian_var.get(), self.segments_var.get(), self.mcs_var.get()]

        # set up preview image variables
        prev = self.program.handle(ButtonPress.PREVIEW, args=args)
        self.preview_frame = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(prev, cv2.COLOR_BGR2RGB)).resize(shape)) # type: ignore

        # dataset name variable
        self.name = StringVar(mainframe, self.parent.program.dataset_name)

        # set up preview panel
        preview_panel = Canvas(mainframe)
        preview_panel.pack(side="left", expand=True, fill="both")

        # create labels
        og_frame = ttk.Label(preview_panel, image=self.original_frame)
        og_frame_lbl = ttk.Label(preview_panel, text="Original")
        self.preview_img_display = ttk.Label(preview_panel, image=self.preview_frame)
        preview_lbl = ttk.Label(preview_panel, text="Preview")
        
        # pack inside preview panel
        og_frame_lbl.pack(side="top", expand=True)
        og_frame.pack(side="top", expand=True)
        preview_lbl.pack(side="top", expand=True)
        self.preview_img_display.pack(side="top", expand=True)
        self.preview_panel = preview_panel

        # set up param panel
        param_panel = Canvas(mainframe)
        param_panel.pack(side="right", expand=True, fill="y")

        # create ctrl_panel and color_panel
        ctrl_panel = Canvas(param_panel)
        ctrl_panel.pack(side="bottom", expand=True, fill="y")
        color_panel = Canvas(param_panel)
        color_panel.pack(side="top", expand=True)

        # set up color panel

        # lower eyedropper panel
        lower_panel = Canvas(color_panel)
        lower_panel.pack(side="top", expand=True)

        lower_eyedropper_panel = Canvas(lower_panel)
        lower_eyedropper_panel.pack(side="top", expand=True)
        lower_lbl = ttk.Label(lower_eyedropper_panel, text="lower threshold: ")
        lower_button = ttk.Button(lower_eyedropper_panel, text="select from image", command=self.select_lower_threshold)
        lower_str = ttk.Label(lower_eyedropper_panel, textvariable=self.lower_var)

        lower_lbl.pack(side="left", expand=True)
        lower_str.pack(side="left", expand=True)
        lower_button.pack(side="left", expand=True)

        # lower text panel
        lower_text_panel = Canvas(lower_panel)
        lower_text_panel.pack(side="bottom", expand=True)
        lower_r_lbl = ttk.Label(lower_text_panel, text="red: ")
        lower_g_lbl = ttk.Label(lower_text_panel, text="green: ")
        lower_b_lbl = ttk.Label(lower_text_panel, text="blue: ")
        self.lower_r = IntVar(mainframe, value=self.program.lower_bound[2])
        self.lower_g = IntVar(mainframe, value=self.program.lower_bound[1])
        self.lower_b = IntVar(mainframe, value=self.program.lower_bound[0])
        lower_r_spin = ttk.Spinbox(lower_text_panel, from_=0, to=255, textvariable=self.lower_r, increment=1)
        lower_g_spin = ttk.Spinbox(lower_text_panel, from_=0, to=255, textvariable=self.lower_g, increment=1)
        lower_b_spin = ttk.Spinbox(lower_text_panel, from_=0, to=255, textvariable=self.lower_b, increment=1)
        lower_spin_btn = ttk.Button(lower_text_panel, text="update lower threshold", command=self.update_lower_spin)

        lower_r_lbl.pack(side="left", expand=True)
        lower_r_spin.pack(side="left", expand=True)
        lower_g_lbl.pack(side="left", expand=True)
        lower_g_spin.pack(side="left", expand=True)
        lower_b_lbl.pack(side="left", expand=True)
        lower_b_spin.pack(side="left", expand=True)
        lower_spin_btn.pack(side="right", expand=True)

        # lower eyedropper panel
        upper_panel = Canvas(color_panel)
        upper_panel.pack(side="bottom", expand=True)

        upper_eyedropper_panel = Canvas(upper_panel)
        upper_eyedropper_panel.pack(side="top", expand=True)
        upper_lbl = ttk.Label(upper_eyedropper_panel, text="upper threshold: ")
        upper_button = ttk.Button(upper_eyedropper_panel, text="select from image", command=self.select_upper_threshold)
        upper_str = ttk.Label(upper_eyedropper_panel, textvariable=self.upper_var)

        upper_lbl.pack(side="left", expand=True)
        upper_str.pack(side="left", expand=True)
        upper_button.pack(side="left", expand=True)

        # upper text panel
        upper_text_panel = Canvas(upper_panel)
        upper_text_panel.pack(side="bottom", expand=True)
        upper_r_lbl = ttk.Label(upper_text_panel, text="red: ")
        upper_g_lbl = ttk.Label(upper_text_panel, text="green: ")
        upper_b_lbl = ttk.Label(upper_text_panel, text="blue: ")
        self.upper_r = IntVar(mainframe, value=self.program.upper_bound[2])
        self.upper_g = IntVar(mainframe, value=self.program.upper_bound[1])
        self.upper_b = IntVar(mainframe, value=self.program.upper_bound[0])
        upper_r_spin = ttk.Spinbox(upper_text_panel, from_=0, to=255, textvariable=self.upper_r, increment=1)
        upper_g_spin = ttk.Spinbox(upper_text_panel, from_=0, to=255, textvariable=self.upper_g, increment=1)
        upper_b_spin = ttk.Spinbox(upper_text_panel, from_=0, to=255, textvariable=self.upper_b, increment=1)
        upper_spin_btn = ttk.Button(upper_text_panel, text="update upper threshold", command=self.update_upper_spin)

        upper_r_lbl.pack(side="left", expand=True)
        upper_r_spin.pack(side="left", expand=True)
        upper_g_lbl.pack(side="left", expand=True)
        upper_g_spin.pack(side="left", expand=True)
        upper_b_lbl.pack(side="left", expand=True)
        upper_b_spin.pack(side="left", expand=True)
        upper_spin_btn.pack(side="right", expand=True)

        # name panel
        name_panel = Canvas(ctrl_panel)
        name_panel.pack(side="top", expand=True)
        name_entry = ttk.Entry(name_panel, textvariable=self.name)
        name_lbl = ttk.Label(name_panel, text="dataset name")
        name_btn = ttk.Button(name_panel, text="change name", command=self.change_name)

        name_lbl.pack(side="left", expand=True)
        name_entry.pack(side="left", expand=True)
        name_btn.pack(side="left", expand=True)

        # set up ctrl panel with entry panel and btn panel
        vars_panel = Canvas(ctrl_panel)
        vars_panel.pack(side="top", fill="y")
        lbls_panel = Canvas(vars_panel)
        entry_panel = Canvas(vars_panel)
        lbls_panel.pack(side="left", expand=True, fill="y")
        entry_panel.pack(side="right", expand=True, fill="y")
        btn_panel = Canvas(ctrl_panel)
        btn_panel.pack(side="bottom", expand=True, fill="both")

        # label panel
        brightness_lbl = ttk.Label(lbls_panel, text="brightness")
        gaussian_lbl = ttk.Label(lbls_panel, text="blur")
        contrast_lbl = ttk.Label(lbls_panel, text="contrast")
        epsilon_lbl = ttk.Label(lbls_panel, text="accuracy")
        segments_lbl = ttk.Label(lbls_panel, text="num segments")
        mcs_lbl = ttk.Label(lbls_panel, text="minimum pixels per cluster")

        brightness_lbl.pack(side="top", expand=True)
        gaussian_lbl.pack(side="top", expand=True)
        contrast_lbl.pack(side="top", expand=True)
        epsilon_lbl.pack(side="top", expand=True)
        segments_lbl.pack(side="top", expand=True)
        mcs_lbl.pack(side="top", expand=True)

        # entry panel
        brightness_entry = ttk.Entry(entry_panel, textvariable=self.brightness_var)
        gaussian_entry = ttk.Entry(entry_panel, textvariable=self.gaussian_var)
        contrast_entry = ttk.Entry(entry_panel, textvariable=self.contrast_var)
        epsilon_entry = ttk.Entry(entry_panel, textvariable=self.epsilon_var)
        segments_entry = ttk.Entry(entry_panel, textvariable=self.segments_var)
        mcs_entry = ttk.Entry(entry_panel, textvariable=self.mcs_var)

        brightness_entry.pack(side="top", expand=True)
        gaussian_entry.pack(side="top", expand=True)
        contrast_entry.pack(side="top", expand=True)
        epsilon_entry.pack(side="top", expand=True)
        segments_entry.pack(side="top", expand=True)
        mcs_entry.pack(side="top", expand=True)


        # button panel
        preview_btn = ttk.Button(btn_panel, text="Preview", command=self.preview)
        continue_btn = ttk.Button(btn_panel, text="Continue", command=self.cont)
        change_save_btn = ttk.Button(btn_panel, text="Change Save Directory", command=self.change_save_directory)
        start_over_btn = ttk.Button(btn_panel, text="Main Menu", command=self.main_menu)
        exit_btn = ttk.Button(btn_panel, text="Exit", command=self.parent.finish)

        preview_btn.pack(side="left", expand=True)
        continue_btn.pack(side="left", expand=True)
        change_save_btn.pack(side="right", expand=True)
        start_over_btn.pack(side="right", expand=True)
        exit_btn.pack(side="right", expand=True)

        # keybinds
        root.bind("<Return>", self.cont)

        # set attribute
        self.mainframe = mainframe        

    def get_image_size(self, img):
        """ Gets an appropriate image size for display purposes.

        Parameters
        ----------
        img : Mat
            The image to resize

        Returns
        -------
        new_width, new_height : int, int
            The new image size as (width, height).

        """

        # get screen dimensions
        screen_height = self.root.winfo_screenheight()
        screen_width = self.root.winfo_screenwidth()

        # calculate max dimensions
        max_width = (1 / 4) * screen_width
        max_height = (1 / 4) * screen_height

        # get image dimensions
        img_shape = img.shape
        img_height = img_shape[0]
        img_width = img_shape[1]

        # if oriented vertical (portait), the limiting variable is height.
        if img_height > img_width:
            # scale img to max height
            new_height = max_height
            new_width = max_height * (img_width / img_height)

            # return new size
            return (int(new_width), int(new_height))

        # if oriented horizontal (landscape), the limiting variable is width
        elif img_height < img_width:
            # scale img to max width
            new_width = max_width
            new_height = max_width * (img_height / img_width)

            # return new size
            return (int(new_width), int(new_height))

        # if square, return the smaller of the two maxes. If maxes are the same, return both.
        elif max_width < max_height:
            return (int(max_width), int(max_width))
        elif max_height < max_width:
            return (int(max_height), int(max_height))
        else:
            return (int(max_width), int(max_height))
        
    def change_name(self):
        """ Handles the change_name button press and passes the entry to the program.

        Returns
        -------
        None.

        """

        self.program.handle(ButtonPress.CHANGE_NAME, args=[self.name.get()])

    def select_lower_threshold(self):
        """ Opens a Matplotlib window for lower threshold color selection and passes the selected
        value to the program.

        Returns
        -------
        None.

        """        

        # open Matplotlib window
        threshold = self.select_color(self.img)

        # skip if nothing was selected
        if type(threshold) == None:
            return

        # pass to program
        self.update_color(threshold, True)
        # update GUI value
        self.lower_var.set(self.parent.bound_to_str(threshold))
    
    def select_upper_threshold(self):
        """ Opens a Matplotlib window for upper threshold color selection and passes the selected
        value to the program.

        Returns
        -------
        None.

        """        

        # open Matplotlib window
        threshold = self.select_color(self.img)

        # skip if nothing was selected
        if type(threshold) == None:
            return

        # pass to program
        self.update_color(threshold, False)
        # update GUI value
        self.upper_var.set(self.parent.bound_to_str(threshold))
        
    def update_lower_spin(self):
        """ Updates the threshold based on manual lower threshold entry.
        
        Returns
        -------
        None.

        """
                
        # get values
        bgr = [self.lower_b.get(), self.lower_g.get(), self.lower_r.get()]
        self.update_color(bgr, True)
        # update GUI value
        self.lower_var.set(self.parent.bound_to_str(bgr))
    
    def update_upper_spin(self):
        """ Updates the threshold based on manual upper threshold entry.
        
        Returns
        -------
        None.

        """
        
        # update values
        bgr = [self.upper_b.get(), self.upper_g.get(), self.upper_r.get()]
        self.update_color(bgr, False)
        # update GUI value
        self.upper_var.set(self.parent.bound_to_str(bgr))

    def update_color(self, bgr, is_lower):
        """ Handles passing an updated BGR threshold to the back end program.

        Parameters
        ----------
        bgr : list
            3x1 list of 3 values between 0 and 255, inclusive.
        is_lower : bool
            True if the passed bgr value is a lower threshold, false if its the upper threshold.

        Returns
        -------
        None.

        """

        # initialize list        
        new_bgr = []

        # check all three values
        for c in bgr:
            if not str(c).isnumeric():
                raise TypeError("BGR threshold must be a numeric value.")
            elif c >= 0 and c <= 255:
                new_bgr.append(int(c))
            else:
                raise ValueError("BGR values must be between 0 and 255, inclusive. Value " + int(c) + " is out of range.")
        
        # check that there are the correct number of items
        if len(new_bgr) != 3:
            raise ValueError("BGR must have 3 values. Length " + len(new_bgr) + " is invalid.")
        
        # create args
        args = [is_lower, new_bgr]
        # pass to back end
        self.program.handle(ButtonPress.CHANGE_COLOR, args=args)

    def cont(self, *kwargs):
        """ Passes values to the back end and returns to parent for state progression.

        Returns
        -------
        None.

        """

        # create args
        args = [self.epsilon_var.get(), self.contrast_var.get(), self.brightness_var.get(), self.gaussian_var.get(), self.segments_var.get(), self.mcs_var.get()]
        # pass to back end and return to parent
        self.program.handle(ButtonPress.CONTINUE, args=args)
        self.parent.cont()

    def change_save_directory(self):
        """ Handles the change save button press and returns to parent.

        Returns
        -------
        None.

        """

        self.program.handle(ButtonPress.CHANGE_SAVE)
        self.parent.cont()

    def main_menu(self):
        """ Handles the start over button press and returns to parent.

        Returns
        -------
        None.

        """

        self.program.handle(ButtonPress.START_OVER)
        self.parent.cont()

    def get_preview_img_original(self):
        """ Gets the original image from the first file in the program's list of filenames.

        Returns
        -------
        img : Mat
            The first image in the program's list of filenames OR the first frame of the first
            video in the program's list of filenames (depends on the first file's file type).

        Raises
        ------
        ValueError
            If there are no filenames associated with the back end program.
        TypeError
            If the passed file is not an accepted type or is a broken video.

        """     

        # define accepted image types   
        accepted_imgs = [".png", ".jpg", ".tif"]
        
        # check that there are filenames associated with the back end.
        if len(self.parent.program.filenames) <= 0:
            raise ValueError("Cannot proceed with no filenames.")
        
        # get the first file
        f = self.parent.program.filenames[0]
        is_video = False

        # determine if the file is a video or an image
        if ".avi" in f:
            is_video = True

        # check that the file is of an accepted type
        acceptable = False
        if is_video:
            acceptable = True

        for t in accepted_imgs:
            if t in f:
                acceptable = True

        # file is not an acceptable format
        if not acceptable:    
            # not a PNG, JPG, or TIF
            raise TypeError("File <" + f + "> is not an appropriate format.")
        
        # get first frame of the video
        if is_video:
            # open video capture
            vid = cv2.VideoCapture(f)
            # read frame
            ret, frame = vid.read()

            # return the img
            if ret:
                img = frame.copy()
                vid.release()
                return img
            # broken video
            else:
                vid.release()
                raise TypeError("Video file <" + f + "> is not a valid video.")
            
        # get the image
        else:
            img = cv2.imread(f)
            return img

    def select_color(self, img):
        """ Opens a matplotlib window and allows the user to select an RGB value from the image.

        Parameters
        ----------
        img : Mat
            The image to choose a color from.

        Returns
        -------
        list
            A 3x1 list of ints between 0 and 255, inclusive representing the BGR color value
            selected by the user.

        """

        # set up window
        fig, ax = plt.subplots()
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # select point
        point = plt.ginput(n=1)[0]

        # close window
        plt.close(fig)

        # get BGR value from the img
        bgr = img[int(point[1])][int(point[0])]

        # convert to ints
        bgr = [int(bgr[0]), int(bgr[1]), int(bgr[2])]

        # return the bgr value
        return bgr

    def preview(self):
        """ Updates the preview image with the current preprocessing parameters.

        Returns
        -------
        None.

        """

        # expected args format:        
        # img = args[0]
        # self.eps = args[1]
        # self.contrast = args[2]
        # self.brightness = args[3]
        # self.gaussian = args[4]
        # self.segments = args[5]
        # self.mcs = args[6]
        args = [self.img, self.epsilon_var.get(), self.contrast_var.get(), self.brightness_var.get(), self.gaussian_var.get(), self.segments_var.get(), self.mcs_var.get()]

        # get preview image from back end
        prev = self.program.handle(ButtonPress.PREVIEW, args=args)

        # reset label widget
        self.preview_img_display.destroy()

        # convert image to PhotoImage
        rgb = cv2.cvtColor(prev, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb).resize(self.get_image_size(rgb)) # type: ignore
        self.preview_frame = ImageTk.PhotoImage(pil_img)
        
        # create Label and format
        self.preview_img_display = ttk.Label(self.preview_panel, image=self.preview_frame)
        self.preview_img_display.pack(side="top", expand=True)


class Processing:
    """ Window for processing the files on the back end.

    Attributes
    ----------
    parent : TrainingDataGenerator
        The parent window.
    root : Tk
        The top-level Tkinter widget associated with this GUI.
    program : Program
        The back-end state machine handling the commands passed from the GUI.
    mainframe : ttk.Frame
        The window associated with this class.
    progress : ttk.ProgressBar
        Tkinter widget indicating approximate completion by the back end in file processing.
    current_file : StringVar
        Variable containing the current file being process.
    current_file_lbl : ttk.Label
        Tkinter widget displaying the label showing the current file.
    
    """ 

    def __init__(self, parent, root):
        """ Initializes the preview window.

        Parameters
        ----------
        parent : TrainingDataGenerator
            The top level GUI window.
        root : Tk
            Top level widget for this GUI.

        Returns
        -------
        None.

        """

        # set attributes
        self.parent = parent
        self.root = root
        self.program = self.parent.program

        # set up the window
        mainframe = ttk.Frame(self.root, padding="3 3 12 12")
        self.mainframe = mainframe
        mainframe.pack(expand=True, fill="both")

        # set up start panel
        start_panel = Canvas(mainframe)
        start_panel.pack(side="top", expand=True, fill="both")
        instructions = "Process Files"
        instructions_lbl = ttk.Label(start_panel, text=instructions)
        start = ttk.Button(start_panel, text="start", command=self.process)
        instructions_lbl.pack(side="left", expand=True)
        start.pack(side="right", expand=True)

        # set up progress bar
        progress_panel = Canvas(mainframe)
        progress_panel.pack(side="top", expand=True, fill="both")
        self.progress = ttk.Progressbar(progress_panel, orient=HORIZONTAL, length=200, mode="determinate")
        self.progress.pack(expand=True)

        # set up file panels
        file_panel = Canvas(mainframe)
        file_panel.pack(side="bottom", expand=True, fill="y")
        self.current_file = StringVar(mainframe, self.program.filenames[0])
        self.current_file_lbl = ttk.Label(file_panel, textvariable=self.current_file)
        processing_lbl = ttk.Label(file_panel, text="processing file: ")
        processing_lbl.pack(side="left", expand=True)
        self.current_file_lbl.pack(side="right", expand=True)

    def process(self):
        """ Processes all the files upon user hitting process.

        Returns
        -------
        None.

        """

        counter = 0
        # get step weight for progress bar
        step_weight = 200 / len(self.program.filenames)

        # process each file
        for f in self.program.filenames:
            # set current file
            self.current_file.set(f)
            # get save directory
            save_dir = self.program.save_dir

            # generate datasets
            if len(self.program.filenames) == 1:
                labeler.generate_dataset(f, self.program.dataset_name, "labels.json", lower_bound=self.program.lower_bound, upper_bound=self.program.upper_bound, eps=self.program.eps, min_cluster_size=self.program.mcs, contrast=self.program.contrast, brightness=self.program.brightness, gaussian=self.program.gaussian, segments=self.program.segments)
            else:
                labeler.generate_dataset(f, self.program.dataset_name + str(counter), "labels.json", lower_bound=self.program.lower_bound, upper_bound=self.program.upper_bound, eps=self.program.eps, min_cluster_size=self.program.mcs, contrast=self.program.contrast, brightness=self.program.brightness, gaussian=self.program.gaussian, segments=self.program.segments)

            # get previews
            if ".avi" in f:
                labeler.generate_video(f, save_dir + "\\" + self.program.dataset_name + str(counter) + ".mp4", lower_bound=self.program.lower_bound, upper_bound=self.program.upper_bound, eps=self.program.eps, min_cluster_size=self.program.mcs, contrast=self.program.contrast, brightness=self.program.brightness, gaussian=self.program.gaussian, segments=self.program.segments)
            else:
                labeler.get_clusters(f, self.program.lower_bound, self.program.upper_bound, self.program.eps, self.program.mcs, self.program.contrast, self.program.brightness, self.program.gaussian, self.program.segments, False, labeler.INTERMEDIATE_DESTINATION, save_dir + "\\" + self.program.dataset_name + str(counter) + ".png", False, True)

            # step progress
            self.progress.step(step_weight)
            # increase counter
            counter += 1
        
        # return to parent and back end
        self.program.handle(ButtonPress.CONTINUE)
        self.parent.cont()


class Previewing:
    """ Window for previewing the processed files.

    Attributes
    ----------
    parent : TrainingDataGenerator
        The parent window.
    root : Tk
        The top-level Tkinter widget associated with this GUI.
    program : Program
        The back-end state machine handling the commands passed from the GUI.
    mainframe : ttk.Frame
        The window associated with this class.
    file_list : list
        List of files in the save directory.
    current_file : str
        The file currently being previewed.
    media_window : Canvas
        The canvas containing media being previewed.
    photo : PhotoImage
        The image being previewed.

    """ 

    def __init__(self, parent, root):
        """ Initializes the preview window.

        Parameters
        ----------
        parent : TrainingDataGenerator
            The parent GUI program managing this one.
        root : Tk
            The top level Tkinter widget managing this GUI.

        """

        # set attributes        
        self.parent = parent
        self.root = root
        self.program = parent.program

        # unbind config
        root.unbind("<Configure>")

        # set up window
        mainframe = ttk.Frame(self.root, padding="3 3 12 12")
        mainframe.pack(expand=True, fill="both")

        # get file information
        self.file_list = os.listdir(self.program.save_dir)
        self.current_file = self.file_list[0]

        # set up media panel
        media_panel = Canvas(mainframe)

        # set up instructions
        instructions = "Preview Files"
        instructions_lbl = ttk.Label(media_panel, text=instructions)
        instructions_lbl.pack(side="top", expand=True)

        # set up media and control panel
        media = Canvas(media_panel, height=300, width=300)
        self.media_window = media
        media_ctrl_panel = Canvas(media_panel)
        media_panel.pack(side="left", fill="both", expand=True)
        media.pack(side="top", expand=True)
        media_ctrl_panel.pack(side="bottom", expand=True)

        # initialize media attributes and get first preview
        self.photo = None
        f = self.program.save_dir + "/" + self.current_file
        self.load_media(f)

        # set up other panels
        info_panel = Canvas(mainframe)
        vars_panel = Canvas(info_panel)
        lbl_panel = Canvas(vars_panel)
        val_panel = Canvas(vars_panel)
        ctrl_panel = Canvas(info_panel)

        # organize panels
        info_panel.pack(side="right", expand=True, fill="both")
        vars_panel.pack(side="top", expand=True)
        lbl_panel.pack(side="left", expand=True)
        val_panel.pack(side="right", expand=True)
        ctrl_panel.pack(side="bottom", expand=True, fill="x")

        # set up media control panel buttons
        prev_btn = ttk.Button(media_ctrl_panel, text="previous file", command=self.get_prev)
        next_btn = ttk.Button(media_ctrl_panel, text="next file", command=self.get_next)
        prev_btn.pack(side="left", expand=True)
        next_btn.pack(side="right", expand=True)

        # set up control panel buttons
        upload_btn = ttk.Button(ctrl_panel, text="upload", command=self.upload)
        restart_btn = ttk.Button(ctrl_panel, text="start over", command=self.start_over)
        exit_btn = ttk.Button(ctrl_panel, text="exit", command=self.parent.finish)
        upload_btn.pack(side="left", expand=True)
        restart_btn.pack(side="left", expand=True)
        exit_btn.pack(side="left", expand=True)

        # set up label panel
        lower_lbl = ttk.Label(lbl_panel, text="lower threshold: ")
        upper_lbl = ttk.Label(lbl_panel, text="upper threshold: ")
        brightness_lbl = ttk.Label(lbl_panel, text="brightness: ")
        gaussian_lbl = ttk.Label(lbl_panel, text="blur: ")
        contrast_lbl = ttk.Label(lbl_panel, text="contrast: ")
        epsilon_lbl = ttk.Label(lbl_panel, text="accuracy: ")
        segments_lbl = ttk.Label(lbl_panel, text="num segments: ")
        mcs_lbl = ttk.Label(lbl_panel, text="minimum pixels per cluster: ")

        # organize label panel
        lower_lbl.pack(side="top", expand=True)
        upper_lbl.pack(side="top", expand=True)
        brightness_lbl.pack(side="top", expand=True)
        gaussian_lbl.pack(side="top", expand=True)
        contrast_lbl.pack(side="top", expand=True)
        epsilon_lbl.pack(side="top", expand=True)
        segments_lbl.pack(side="top", expand=True)
        mcs_lbl.pack(side="top", expand=True)

        # set up value panel
        lower_val = ttk.Label(val_panel, text=self.parent.bound_to_str(self.program.lower_bound))
        upper_val = ttk.Label(val_panel, text=self.parent.bound_to_str(self.program.upper_bound))
        brightness_val = ttk.Label(val_panel, text=str(self.program.brightness))
        gaussian_val = ttk.Label(val_panel, text=str(self.program.gaussian))
        contrast_val = ttk.Label(val_panel, text=str(self.program.contrast))
        epsilon_val = ttk.Label(val_panel, text=str(self.program.eps))
        segments_val = ttk.Label(val_panel, text=str(self.program.segments))
        mcs_val = ttk.Label(val_panel, text=str(self.program.mcs))

        # organize value panel
        lower_val.pack(side="top", expand=True)
        upper_val.pack(side="top", expand=True)
        brightness_val.pack(side="top", expand=True)
        gaussian_val.pack(side="top", expand=True)
        contrast_val.pack(side="top", expand=True)
        epsilon_val.pack(side="top", expand=True)
        segments_val.pack(side="top", expand=True)
        mcs_val.pack(side="top", expand=True)

        # set attribute
        self.mainframe = mainframe

    def upload(self):
        """ Moves program forward to the next state and returns to parent.

        Returns
        -------
        None.

        """

        self.program.handle(ButtonPress.UPLOAD)
        self.parent.cont()

    def get_next(self):
        """ Gets the next file in the directory.

        Returns
        -------
        None.

        """

        # get the position in the list
        index = self.file_list.index(self.current_file)
        new_index = index

        # calculate the new index (if at end of list, reset to 0, otherwise, increment)
        if index == len(self.file_list) - 1:
            new_index = 0
        else:
            new_index = index + 1

        # get the new file name and load the media
        self.current_file = self.file_list[new_index]
        self.load_media(self.program.save_dir + "/" + self.file_list[new_index])
        
    def get_prev(self):
        """ Gets the previous file in the directory.

        Returns
        -------
        None.

        """
                
        # get the position in the list
        index = self.file_list.index(self.current_file)
        new_index = index

        # calculate the new index (if at start of list, reset to end; otherwise, decrement)
        if index == 0:
            new_index = len(self.file_list) - 1
        else:
            new_index = index - 1

        # get the new file name and load the media
        self.current_file = self.file_list[new_index]
        self.load_media(self.program.save_dir + "/" + self.file_list[new_index])

    def start_over(self):
        """ Handles a user request to start over.

        Returns
        -------
        None.

        """

        self.program.handle(ButtonPress.START_OVER)
        self.parent.cont()

    def load_media(self, f):
        """ Loads a requested file in the media window.

        Parameters
        ----------
        f : str
            Path to the file to load.

        Returns
        -------
        None.

        """
        
        # destroy children widgets 
        for c in self.media_window.winfo_children():
            c.destroy()

        # determine if video or image
        if ".mp4" in f or ".avi" in f:
            # set up video player
            videoplayer = tktv(master=self.media_window, scaled=True)
            # load video
            videoplayer.load(f)
            videoplayer.pack(expand=True, fill="both")
            videoplayer.play() # play the video

        else:
            # read the image
            img = cv2.imread(f)
            # calculate image size
            new_size = self.get_image_size(img)
            # read the image as an Image and resize
            img = Image.open(f).resize(new_size) # type: ignore
            # set attributes and organize window
            self.photo = ImageTk.PhotoImage(img)
            self.media_window.create_image((0,0), image=self.photo, anchor='nw')

    def get_image_size(self, img):
        """ Gets new image dimensions based on the screen size.

        Parameters
        ----------
        img : Mat
            Image to resize to desired ratio.

        Returns
        -------
        new_width : int
            New width for the image.
        new_height : int
            New height for the image.

        """

        # set max values
        max_width = 300
        max_height = 300

        # get default image dimensions
        img_shape = img.shape
        img_height = img_shape[0]
        img_width = img_shape[1]

        # switch between different orientations 
        if img_height > img_width:
            # oriented vertical
            # limiting var is height
            # scale img to max height
            new_height = max_height
            new_width = max_height * (img_width / img_height)
            return (int(new_width), int(new_height))

        elif img_height < img_width:
            # oriented landscape
            # limited by width
            new_width = max_width
            new_height = max_width * (img_height / img_width)
            return (int(new_width), int(new_height))

        else:
            # square
            return (int(max_width), int(max_height))


class Uploading:
    """ Window for uploading to Roboflow.

    Attributes
    ----------
    parent : TrainingDataGenerator
        The parent window.
    root : Tk
        The top-level Tkinter widget associated with this GUI.
    program : Program
        The back-end state machine handling the commands passed from the GUI.
    mainframe : ttk.Frame
        The window associated with this class.
    project_id : StringVar
        The user-defined project to upload to.
    batch : StringVar
        The name for the upload.
    api : StringVar
        The private api key for upload.

    """ 

    def __init__(self, parent, root):
        """ Initializes the upload window.

        Parameters
        ----------
        parent : TrainingDataGenerator
            The top level GUI program.
        root : Tk
            The top level Widget managing this GUI.
        
        Returns
        -------
        None.
        
        """

        # set attributes
        self.parent = parent
        self.root = root
        self.program = parent.program

        # set up window
        mainframe = ttk.Frame(self.root, padding="3 3 12 12")
        mainframe.pack(expand=True, fill="both")

        # create instructions
        instructions = "To upload this dataset to Roboflow, please enter your workspace, project ID, a batch name for the upload (by default the dataset name), and your API key."

        additional_instructions = "Your API Key is a random string of characters generated automatically by Roboflow. To obtain the API key, go to your Roboflow dashboard. Click SETTINGS. The API key will be listed under settings/<your workspace name>/Roboflow API/private API key."

        # set up information panel
        instructions_label = ttk.Label(mainframe, text=instructions)
        instructions_label.pack(side="top", expand=True)
        info_panel = Canvas(mainframe)
        info_panel.pack(side="bottom", expand=True, fill="both")

        # set up overview panel
        overview_panel = Canvas(info_panel)
        overview_panel.pack(side="left", expand=True, fill="both")

        # set up directory and control panels
        dir_panel = Canvas(overview_panel)
        ctrl_panel = Canvas(overview_panel)
        dir_panel.pack(side="top", fill="both", expand=True)
        ctrl_panel.pack(side="bottom", fill="x", expand=True)

        # set up roboflow panel
        rf_panel = Canvas(info_panel)
        rf_panel.pack(side="right", expand=True, fill="both")

        # create directory panel widgets
        dir_lbl = ttk.Label(dir_panel, text="Selected directory: ")
        dir_lbl.pack(side="left", expand=True)
        dir_entry = ttk.Label(dir_panel, textvariable=self.program.save_dir)
        dir_entry.pack(side="left", expand=True)
        dir_btn = ttk.Button(dir_panel, text="change source directory", command=self.change_src)
        dir_btn.pack(side="right", expand=True)

        # create control panel widgets
        restart_btn = ttk.Button(ctrl_panel, text="start over", command=self.start_over)
        exit_btn = ttk.Button(ctrl_panel, text="exit", command=self.parent.finish)
        restart_btn.pack(side="left", expand=True)
        exit_btn.pack(side="right", expand=True)

        # set up attribute variables
        project_id_var = StringVar(mainframe)
        batch_name_var = StringVar(mainframe, value=self.program.dataset_name)
        api_key_var = StringVar(mainframe)
        self.project_id = project_id_var
        self.batch = batch_name_var
        self.api = api_key_var

        # set up rows for panel
        id_panel = Canvas(rf_panel)
        id_panel.pack(side="top", expand=True)

        # set up ID widgets
        id_lbl = ttk.Label(id_panel, text="Project ID: ")
        id_lbl.pack(side="left", expand=True)
        id_entry = ttk.Entry(id_panel, textvariable=project_id_var)
        id_entry.pack(side="right", expand=True)

        # set up batch row
        batch_panel = Canvas(rf_panel)
        batch_panel.pack(side="top", expand=True)

        # set up batch widgets
        batch_lbl = ttk.Label(batch_panel, text="Batch name: ")
        batch_lbl.pack(side="left", expand=True)
        batch_entry = ttk.Entry(batch_panel, textvariable=batch_name_var)
        batch_entry.pack(side="right", expand=True)

        # set up api key row
        key_panel = Canvas(rf_panel)
        key_panel.pack(side="top", expand=True)

        # set up api key widgets
        key_lbl = ttk.Label(key_panel, text="API Key: ")
        key_lbl.pack(side="left", expand=True)
        key_entry = ttk.Entry(key_panel, textvariable=api_key_var, show='*')
        key_entry.pack(side="right", expand=True)

        # create additional roboflow panel widgets
        upload_btn = ttk.Button(rf_panel, text="upload", command=self.upload)
        upload_btn.pack(side="bottom", expand=True)
        additional_instructions_lbl = ttk.Label(rf_panel, text=additional_instructions)
        additional_instructions_lbl.pack(side="bottom", expand=True)

        # set attribute
        self.mainframe = mainframe
    
    def start_over(self):
        """ Handles user request to start over.

        Returns
        -------
        None.

        """

        self.program.handle(ButtonPress.START_OVER)
        self.parent.cont()

    def change_src(self):
        """ Handles user request to change source directory.

        Returns
        -------
        None.

        """

        return

    def upload(self):
        """ Uploads dataset to Roboflow.

        Returns
        -------
        None.

        """

        # expected format:        
        # api_key = args[0]
        # imgs = args[1]
        # annotations = args[2]
        # project_id = args[3]
        # batch_name = args[4]

        # attributes
        # self.project_id = project_id_var
        # self.batch = batch_name_var
        # self.api = api_key_var

        # get images folder
        destination_folder = "model/datasets/" + self.program.dataset_name + "/train/images"

        # get path to labels
        folder_loc = destination_folder
        json_loc = "model/datasets/" + self.program.dataset_name + "/train/labels/labels.json"

        # change names
        imgs = folder_loc
        annotations = json_loc

        # create args
        args = [self.api.get(), imgs, annotations, self.project_id.get(), self.batch.get()]
        
        # pass to program and return to parent
        self.program.handle(ButtonPress.UPLOAD, args=args)
        self.parent.cont()


class UploadSuccess:
    """ Window for successful upload.

    Attributes
    ----------
    parent : TrainingDataGenerator
        The parent window.
    root : Tk
        The top-level Tkinter widget associated with this GUI.
    program : Program
        The back-end state machine handling the commands passed from the GUI.
    mainframe : ttk.Frame
        The window associated with this class.

    """ 

    def __init__(self, parent, root):
        """ Initializes the upload success window.

        Parameters
        ----------
        parent : TrainingDataGenerator
            The top level GUI managing this one.
        root : Tk
            The top level Tkinter widget managing this GUI.

        Returns
        -------
        None.
        
        """

        # set attributes
        self.parent = parent
        self.root = root
        self.program = self.parent.program

        # set up window
        mainframe = ttk.Frame(self.root, padding="3 3 12 12")
        mainframe.pack(expand=True, fill="both")

        # create status information
        status = "Upload success!"
        status_lbl = ttk.Label(mainframe, text=status)
        status_lbl.pack(side="top", expand=True)

        # create control panel
        ctrl_panel = Canvas(mainframe)
        ctrl_panel.pack(side="bottom", expand=True)

        # create control panel widgets
        restart_btn = ttk.Button(ctrl_panel, text="start over", command=self.start_over)
        restart_btn.pack(side="left", expand=True)
        exit_btn = ttk.Button(ctrl_panel, text="exit", command=self.parent.finish)
        exit_btn.pack(side="right", expand=True)

        # set attribute
        self.mainframe = mainframe
    
    def start_over(self):
        """ Handles user request to start over.

        Returns
        -------
        None.

        """

        self.program.handle(ButtonPress.START_OVER)
        self.parent.cont()


class UploadFailure:
    """ Window for failed upload.

    Attributes
    ----------
    parent : TrainingDataGenerator
        The parent window.
    root : Tk
        The top-level Tkinter widget associated with this GUI.
    program : Program
        The back-end state machine handling the commands passed from the GUI.
    mainframe : ttk.Frame
        The window associated with this class.

    """ 
        
    def __init__(self, parent, root):
        """ Initializes the upload failure window.

        Parameters
        ----------
        parent : TrainingDataGenerator
            The top level GUI managing this one.
        root : Tk
            The top level Tkinter widget managing this GUI.

        Returns
        -------
        None.
            
        """

        # set attributes
        self.parent = parent
        self.root = root
        self.program = self.parent.program

        # set up window
        mainframe = ttk.Frame(self.root, padding="3 3 12 12")
        mainframe.pack(expand=True, fill="both")

        # create status and error message
        status = "Upload failure :( See error message:"
        status_lbl = ttk.Label(mainframe, text=status)
        status_lbl.pack(side="top", expand=True)
        message = self.program.upload_error
        message_lbl = ttk.Label(mainframe, text=message)
        message_lbl.pack(expand=True)

        # create control panel
        ctrl_panel = Canvas(mainframe)
        ctrl_panel.pack(side="bottom", expand=True)

        # set up control panel widgets
        restart_btn = ttk.Button(ctrl_panel, text="start over", command=self.start_over)
        restart_btn.pack(side="left", expand=True)
        exit_btn = ttk.Button(ctrl_panel, text="exit", command=self.parent.finish)
        exit_btn.pack(side="right", expand=True)
        retry_btn = ttk.Button(ctrl_panel, text="retry", command=self.retry)
        retry_btn.pack(expand=True)

        # set attributes
        self.mainframe = mainframe

    def start_over(self):
        """ Handles user request to start over.

        Returns
        -------
        None.

        """

        self.program.handle(ButtonPress.START_OVER)
        self.parent.cont()

    def retry(self):
        self.program.handle(ButtonPress.RETRY)
        self.parent.cont()

# create top level tkinter widget
root = Tk()
# initialize window
TrainingDataGenerator(root)
# run the main loop until exit
root.mainloop()
