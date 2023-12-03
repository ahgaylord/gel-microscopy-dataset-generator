import unittest
import cv2
from src import trainer_gui

class TestClusterAnalyzer(unittest.TestCase):
    """ Tests the cluster analyzer class and calculates accuracy

    """

    IMG_ONE = "testdocs/fileOne.png"
    IMG_TWO = "testdocs/fileTwo.png"
    IMG_THREE = "testdocs/fileThree.png"
    # IMG_FOUR = "testdocs/fileFour.png"
    # IMG_FIVE = "testdocs/fileFive.png"

    # VID_ONE = "testdocs/vidOne.mp4"
    # VID_TWO = "testdocs/vidTwo.mp4"
    # VID_THREE = "testdocs/vidThree.mp4"

    def __init__(self, methodName: str = "runTest") -> None:
        self.test_one = cv2.imread(self.IMG_ONE)
        self.test_two = cv2.imread(self.IMG_TWO)
        self.test_three = cv2.imread(self.IMG_THREE)
        # self.test_four = cv2.imread(self.IMG_FOUR)
        # self.test_five = cv2.imread(self.IMG_FIVE)

        # self.test_six = cv2.imread(self.VID_ONE)
        # self.test_seven = cv2.imread(self.VID_TWO)
        # self.test_eight = cv2.imread(self.VID_THREE)

        self.program = trainer_gui.Program()

        super().__init__(methodName)

    def load_test_files(self):
        """ Loads test files and saves them

        """
        
        self.test_one = cv2.imread(self.IMG_ONE)
        self.test_two = cv2.imread(self.IMG_TWO)
        self.test_three = cv2.imread(self.IMG_THREE)
        # self.test_four = cv2.imread(self.IMG_FOUR)
        # self.test_five = cv2.imread(self.IMG_FIVE)

        # self.test_six = cv2.imread(self.VID_ONE)
        # self.test_seven = cv2.imread(self.VID_TWO)
        # self.test_eight = cv2.imread(self.VID_THREE)

    def setUp(self) -> None:
        self.program = trainer_gui.Program()
        self.load_test_files()
        return super().setUp()

    def test_handle(self):
        """ Tests the handle method
        """

        self.assertRaises(ValueError.__class__, self.program.handle(-1))
        self.assertRaises(ValueError.__class__, self.program.hanlde(11))

    def test_features(self):
        """ Tests the generate_feature function
        """
        # self.assertEqual()

        self.assertEqual(sum([1, 2, 3]), 6, "Should be 6")

if __name__ == '__main__':
    unittest.main()
