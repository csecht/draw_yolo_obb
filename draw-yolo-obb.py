"""
This script draws oriented bounding boxes on objects in an image.
It saves box coordinates in a text format suitable for training and
fine-tuning of yolo-obb models from the Ultralytics package:
class_index x1 y1 x2 y2 x3 y3 x4 y4. An annotated image with boxes
is also saved for reference.

At startup, an example image is loaded. New images can be loaded using
a button on the YOLO OBB Control window.

There are three ways to draw a box:
Option 1: Click on the image and press 'b' to place a small box near the
top-left corner of the image.
Option 2: Right-click and drag to draw a new box anywhere in the image.
Option 3: Click in an existing box, press 'c' to clone it, then drag the
new box to a new location.

To move and reshape a box: click inside it to activate, then drag to
reposition. Click and drag the dot in the bottom-right corner to resize.
Or use the keyboard, as described below. The active box is highlighted
in red, inactive boxes in green. Only an active box can be manipulated
with the mouse or keyboard.

Use the YOLO OBB Control window to enter a class index for the box to
be drawn. The default class index is 0. Enter an index value for a
different class BEFORE creating a new box. The number entered will be
used for subsequent boxes until changed. The class index number is
displayed in a box's drag corner.

Keyboard manipulation of the active box:
The left and right arrow keys rotate the box in 3 degree increments.
The up and down arrow keys increase or decrease the box size in Linux.
The 'i', 'k', 'j', and 'l' keys move the box up, down, left, and right.
The 'e', 'd', 'f', and 's' keys increase or decrease height and width.
The 'r' key removes the active (red) box.
The 't' key toggles the display of the class index corner circle.

Program control keys:
The up and down arrow keys zoom the image in and out in Windows.
The mouse wheel zooms in and out in Linux.
'Increment' values in the Control window set pixel and angle units
  for keyboard manipulations. Lower values are more precise, but slower.
The 'Esc' key quits program from the control window.
Control-q quits the program from the control window.
The 'X' button in the control window bar quits the program.
The 'Help' button pops up a window of usage information (this docstring).

Clicking the image window's 'X' button will not end the program, it just
redraws the window.

The terminal/console window provides some messaging.
It maybe covered by the main image window that fills most of the screen.
Reposition windows as needed. The control window is always on top.
-- END OF USAGE INFO --
"""

# Standard imports
import sys
import threading
import tkinter as tk

from pathlib import Path
from tkinter import messagebox, filedialog
from tkinter.scrolledtext import ScrolledText
from typing import Optional, Union

# Third party imports
import cv2
import numpy as np

MY_OS = sys.platform[:3]  # 'lin', 'win', or 'dar'.
if MY_OS == 'dar':
    print('macOS is not supported; Linux and Windows are.')
    sys.exit(1)

LABELS_FOLDER = 'labels'  # Default local folder for yolo labels files.
RESULTS_DIR = 'results'


class Box:
    """
    A class to represent a rectangular polygon for use as a YOLO oriented
    bounding box.
    Called from BoxDrawer class to append the box's class index,
    point coordinates, and rotation angle to the boxes list whenever
    a box is drawn or manipulated.
    """

    def __init__(self, class_index=0, points=None, rotation_angle=0, ):
        self.class_index: int = class_index

        # The points argument can be a list or a numpy array.
        self.points = points if points is not None else np.array([(0, 0), (0, 0)])
        self.rotation_angle = rotation_angle  # degrees
        self.center = (0, 0)
        self.width = 0
        self.height = 0
        self.is_active = False  # Tracks if this box is currently active

    def update_properties(self):
        """
        Calculate the center, width, and height of a box; transforms
        the points based on points and rotation angle.
        Called by BoxDrawer.handle_mouse_events(), key events in
        BoxDrawer.draw_box(), and Utility.view_labels().
        The *points* argument is expected to contain either two or four
        points, and cen be either a list or a numpy array.
        """

        img_h, img_w = box_drawer.image_info['h&w']

        # Clip points to ensure they are within image boundaries
        self.points = np.clip(self.points, a_min=[0, 0], a_max=[img_w - 1, img_h - 1])
        self.center = np.mean(self.points, axis=0)

        if len(self.points) == 2:
            # Axially oriented box.
            self.width, self.height = np.abs(self.points[1] - self.points[0])

        elif len(self.points) == 4:
            # Rotated box.
            # Rotate points to align with axes for width and height calculation.
            # The `None` indexing adds a third dimension to the array for
            # `cv2.transform`. The `[0]` indexing extracts the transformed
            # points to re-form a 2D array with new corner coordinates.
            rotation_matrix = cv2.getRotationMatrix2D(
                self.center, self.rotation_angle, scale=1)
            rotated_points = cv2.transform(
                self.points[None, :, :], rotation_matrix)[0]
            self.width = np.ptp(rotated_points[:, 0])  # Peak-to-peak x coord (max - min)
            self.height = np.ptp(rotated_points[:, 1])  # Peak-to-peak y coord (max - min)

    def update_points(self):
        """
        Update the box's points based on its center, width, height, and
        rotation angle.
        Called from check_boundaries(), handle_mouse_events() and
        key events in BoxDrawer class, Utility.view_labels(), and
        when the box properties are updated to recalculate the points
        for the box. This is essential for ensuring the points are
        correctly positioned after any manipulation of the box's
        properties (center, width, height, or rotation angle).
        """

        # Precompute half dimensions.
        half_w, half_h = self.width / 2, self.height / 2

        # Define corners relative to the center.
        corners = np.array([
            [-half_w, -half_h],  # Top-left
            [half_w, -half_h],   # Top-right
            [half_w, half_h],    # Bottom-right
            [-half_w, half_h],   # Bottom-left
        ])

        # Precompute rotation matrix.
        angle_rad = np.radians(self.rotation_angle)
        rotation_matrix = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)]
        ])

        # Rotate and translate corners to pixel coordinates in one step.
        self.points = np.round(corners @ rotation_matrix.T + self.center).astype(int)


class BoxDrawer:
    """
    A Class to draw and manipulate oriented bounding boxes on an OpenCV
    image. Converts and saves resulting box points in a format that can
    be used for yolo-obb model training.
    """
    def __init__(self, image_path: str):

        # Check if image_path exists.
        if not Path(image_path).exists():
            print(f'Image path {image_path} does not exist.'
                  ' Please provide a valid image path.')
            sys.exit(1)

        # Used for threading and synchronization.
        self.stop_loop = threading.Event()
        self.control_lock = threading.RLock()

        self.window_name = ''  # Set in draw_box().

        self.image_array: np.ndarray = cv2.imread(image_path)
        self.display_image = np.array([])  # The active drawing image.
        self.image_path: str = image_path  # The default start image.
        self.image_name: str = Path(image_path).stem  # File name without extension
        self.image_info = {}  # Dictionary populated in update_image_info().

        # Note that list definition here won't pick up files added during program run.
        self.labels_files = [f.name for f in Path(LABELS_FOLDER).glob('*.txt')]

        self.boxes = []  # List to store all box coordinates
        self.active_box = None  # Currently active box
        self.is_dragging_corner = False  # Tracks if a corner is being dragged
        self.is_dragging_box = False  # Tracks if the entire box is being dragged
        self.is_dragging_new_box = False  # Tracks if rt-click drawing a new box.
        self.start_point = (0, 0)  # Starting coordinate to draw a new box.
        self.offset = (0, 0)  # Offset for dragging the entire box
        self.b_box_counter = 0  # Counter for 'b' key presses
        self.current_class_index = 0
        self.show_corner_circle = True

        # Minimum dimension pixel size to prevent disappearing, too small,
        #  and out-of-bounds objects,
        self.min_dim = 12

        # cv2 colors for drawing boxes and text.
        self.cv_color = {
            'white': (255, 255, 255),
            'black': (0, 0, 0),  # Black color for class index text
            'red': (0, 0, 255),  # Red color for active box
            'green': (0, 255, 0),  # Green color for non-active boxes
            'cyan': (255, 255, 0),  # Cyan color for dot in drag corner
            'orange': (0, 159, 230),
            'blue': (255, 0, 0),
        }

        # Attributes for zooming the image in Windows.
        self.zoom_level = 1.0
        self.zoom_step = 0.1  # Same as Linux default
        self.zoom_center = None  # Mouse position when zooming
        self.last_mouse_pos = (0, 0)  # Track mouse position, for Windows.
        self.zoom_key_active = False  # Track if zoom modifier key is pressed

        # Used for platform-specific key handling in handle_keys().
        self.arrow_keys = {}

    def open_image(self) -> None:
        """
        Reads a new image file for drawing.
        Called from new_input() via YoloInfoWindow.config_control_window()
        button command.

        Returns:
            True or False depending on whether input was selected.

        """

        with self.control_lock:
            # Disable window close button 'X'.
            app.protocol('WM_DELETE_WINDOW', lambda: None)

            self.image_path: str = filedialog.askopenfilename(
                # parent=app,
                title='Select input image',
                filetypes=[('JPG', '*.jpg'),
                           ('JPG', '*.jpeg'),
                           ('JPG', '*.JPEG'),
                           ('JPG', '*.JPG'),  # used for iPhone images
                           ('PNG', '*.png'),
                           ('PNG', '*.PNG')
                           ],
            )

            if self.image_path:
                self.image_array = cv2.imread(self.image_path)
                self.boxes.clear()
                new_img_name = Path(self.image_path).name
                app.current_image_name.set(f'Current image: {new_img_name}')
                self.update_image_info()
                cv2.resizeWindow(self.window_name,
                                 self.image_info['window_w'],
                                 self.image_info['window_h'])

                # Check if a labels file exists for the new image.
                self.look_for_labels(self.image_path)

                # Reset to defaults.
                self.b_box_counter = 0
                self.zoom_level = 1.0

            # Re-enable window close button 'X'.
            app.protocol('WM_DELETE_WINDOW', app.on_close)


    def look_for_labels(self, img_path: str) -> None:
        """
        Check if a labels file exists for the image just loaded.
        If a labels file exists, prompt the user to convert the label data

        Args: img_path: The string path to the image file.
        """

        # Ensure the labels folder exists.
        Path(LABELS_FOLDER).mkdir(parents=True, exist_ok=True)

        labels_file = Path(img_path).stem + '.txt'
        with self.control_lock:
            if labels_file in self.labels_files:
                convert_now = messagebox.askyesno(
                    parent=app,
                    title='Convert label data?',
                    detail=f'The image "{Path(img_path).name}"'
                           f' has a labels file in {LABELS_FOLDER}.\n'
                           f'"Yes" draws those boxes on the image.\n')
                if convert_now:
                    # Load the existing label data and draw boxes on the image.
                    labels_path = f'{LABELS_FOLDER}/{labels_file}'
                    try:
                        labels_to_view = app.get_labels(labels_path=labels_path)
                        if labels_to_view:
                            Utility.view_labels(label_data=labels_to_view)
                    except FileNotFoundError as e:
                        print('DEV: From look_for_labels(),'
                              f' no labels file found for {img_path}.\n'
                              f'Error: {e}')

    def update_image_info(self) -> None:
        """
        Assign values to the image_info dict for the current image.
        Called from open_image() and if name == main startup.
        """

        image_name: str = Path(self.image_path).stem
        img_name_and_extension = Path(self.image_path).name
        image_copy: np.ndarray = self.image_array.copy()
        img_h, img_w, _ = self.image_array.shape

        # Need the maximum CV window to be smaller than the screen size,
        #  and comfortably scale-to-fit the largest image dimension.
        win_h = app.winfo_screenheight() * 0.85
        win_w = app.winfo_screenwidth() * 0.85
        display_factor = max(img_h / win_h, img_w / win_w, 1)

        # Ideas for scaling: https://stackoverflow.com/questions/52846474/
        #   how-to-resize-text-for-cv2-puttext-according-to-the-image-object_size-in-opencv-python
        # The conversion factors were empirically determined.
        avg_d = (win_h + win_w) * 0.5
        line_thickness = max(round(avg_d * 0.001 * display_factor), 2)
        font_scale = max(avg_d * 0.0003, 0.4) * display_factor

        # Factors used in cv2.resizeWindow() to scale window to fit each image.
        cv_win_h = round(img_h / display_factor)
        cv_win_w = round(img_w / display_factor)

        self.image_info = {
            'copy': image_copy,
            'h&w': (img_h, img_w),
            'short name': image_name,
            'full name': img_name_and_extension,
            'font scale': font_scale,
            'line thickness': line_thickness,
            # Used by resizeWindow() to perfectly fit the image.
            'window_h': cv_win_h,
            'window_w': cv_win_w,
        }

    def get_circle_radius(self) -> int:
        """
        Returns a scaled radius for a small circle that is relative to
        the font size. Radius is expected the bottom-right corner
        of the active box that holds the class index in a display or
        saved image.
        Returns: An integer value for the circle radius.
        """

        # The font scale multiplication factor was empirically determined.
        return max(int(self.image_info['font scale'] * 15), self.min_dim)

    def set_min_hw(self, min_d: int) -> None:
        """
        Prevents the box from disappearing when height or width
        is getting too small to handle.  Reset to min size dimension.
        Called from the down arrow key event in draw_box() and
        handle_mouse_events() when dragging the box. *min_d* is expected
        to be self.min_dim, for consistency of action.

        Args:
            min_d: The minimum dimension for the box.
        """
        self.active_box.height = max(self.active_box.height, min_d)
        self.active_box.width = max(self.active_box.width, min_d)

    def draw_box(self) -> None:
        """
        Draw rectangular polygons on the image and handle user interactions.
        A while loop keeps the image updated with cv2.imshow and keeps key
        events active for box repositioning, resizing, and initiating
        drawing. Is threaded in main() for live updates and interactions
        with the YoloOBBControl Tk window.
        """

        # CV WINDOW SETUP
        #  Requires cv2.WINDOW_GUI_NORMAL instead of cv2.WINDOW_NORMAL because
        #  WINDOWS_NORMAL runs a Qt thread that expects to be the main loop.
        #  The program runs fine with WINDOWS_NORMAL, but throws a Qt timer
        #  error when closed from app.on_close().
        self.window_name = "View and Edit OBB"
        cv2.namedWindow(self.window_name,cv2.WINDOW_GUI_NORMAL,)
        cv2.resizeWindow(
            self.window_name,
            self.image_info['window_w'],
            self.image_info['window_h'])

        cv2.setMouseCallback(self.window_name,
                             self.handle_mouse_events)

        # BOX DRAWING and KEYBOARD ACTION LOOP
        # The loop provides live drawing updates as long as the threading.Event
        #  is not set. Event is set upon call to app.on_close().
        while not self.stop_loop.is_set():

            # Using a copy allows live drawing of boxes.
            self.display_image = self.image_info['copy'].copy()

            # Zooms window the up/down arrow keys on Windows.
            if MY_OS == 'win' and self.zoom_level != 1.0 and self.zoom_center:
                _h, _w = self.display_image.shape[:2]
                img_h, img_w = self.image_info['h&w']
                scale = self.zoom_level

                # Convert mouse position to image coordinates
                mouse_x, mouse_y = (int(coord * scale) for coord, scale in
                                    zip(self.zoom_center, (_w / img_w, _h / img_h)))

                # Create transformation matrix centered on the mouse position.
                t_matrix = np.array([
                    [scale, 0, (1 - scale) * mouse_x],
                    [0, scale, (1 - scale) * mouse_y]
                ])

                # Apply transformation
                self.display_image = cv2.warpAffine(
                    self.display_image,
                    M=t_matrix,
                    dsize=(_w, _h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT
                )

                # Transform zoomed box coordinates
                for _box in self.boxes:
                    if len(_box.points) == 4:
                        _box.transformed_points = cv2.transform(_box.points[None, :, :], t_matrix)[0]

            for _box in self.boxes:
                if len(_box.points) == 4:
                    if MY_OS == 'win' and self.zoom_level != 1.0 and self.zoom_center:
                        pts = _box.transformed_points
                    else:
                        pts = _box.points

                    # Highlight active box in red, all others in green.
                    color = (self.cv_color['green']
                             if not _box.is_active else self.cv_color['red'])

                    cv2.polylines(img=self.display_image,
                                  pts=[pts],
                                  isClosed=True,
                                  color=color,
                                  thickness=self.image_info['line thickness'],
                                  )

                    # Only draw circle in the bottom-right corner if
                    #  show_corner_circle is True. Always show the class index.
                    # Use different colors for active vs. inactive boxes.
                    # Make radius size relative to image size.
                    if self.show_corner_circle:
                        cv2.circle(
                        self.display_image,
                            center=pts[2],
                            radius=self.get_circle_radius(),
                            color=self.cv_color['cyan'] if _box.is_active else self.cv_color['white'],
                            thickness=cv2.FILLED,
                        )

                    self.put_text_class_index(image=self.display_image,
                                              class_idx=str(_box.class_index),
                                              box_points=pts,)

            cv2.imshow(self.window_name, self.display_image)
            self.handle_keys()

        # stop_loop has been set, so close the window and prepare to exit.
        cv2.destroyWindow(self.window_name)

    def handle_keys(self) -> None:
            """
            Define all the key events for manipulating boxes in the CV window.
            Called from draw_box_frame().
            """

            # Get arrow key press and define key mappings for Windows and Linux.
            # NOTE: key actions are intermittent on macOS (darwin) platform, but
            #  are included for possible future macOS implementation.
            key = cv2.waitKeyEx(1) if MY_OS in ('win', 'dar') else cv2.waitKey(1)
            self.arrow_keys = {
                'left': 2424832 if MY_OS == 'win' else 63234 if MY_OS == 'dar' else 81,
                'up': 2490368 if MY_OS == 'win' else 63232 if MY_OS == 'dar' else 82,
                'right': 2555904 if MY_OS == 'win' else 63235 if MY_OS == 'dar' else 83,
                'down': 2621440 if MY_OS == 'win' else 63233 if MY_OS == 'dar' else 84,
            }

            # Early return if no key pressed or key not in our handlers.
            if key == -1:
                return

            # Pixel increment for box movement, rotation, and resizing.
            #  Larger is faster, but less precise. Best as factor of 180.
            #  Is set in YoloOBBControl class with Radiobuttons.
            incr = app.increment.get()

            # Define a dispatch dictionary to map keys to their handlers.
            key_handlers = {
                ord("b"): self._handle_new_box,
                ord("c"): self._handle_clone_box,
                ord("r"): self._handle_remove_box,
                # ord("h"): Utility.show_help,
                # Arrow keys, platform-specific
                self.arrow_keys['left']: lambda: self._handle_rotate(angle_delta=-incr),
                self.arrow_keys['right']: lambda: self._handle_rotate(angle_delta=incr),
                self.arrow_keys['up']: lambda: self._handle_increase_size(delta=incr),
                self.arrow_keys['down']: lambda: self._handle_decrease_size(delta=incr),
                # Box movement
                ord("i"): lambda: self._handle_move(xy_vector=(0, -incr)),
                ord("k"): lambda: self._handle_move(xy_vector=(0, incr)),
                ord("j"): lambda: self._handle_move(xy_vector=(-incr, 0)),
                ord("l"): lambda: self._handle_move(xy_vector=(incr, 0)),
                # Box sizing
                ord("e"): lambda: self._handle_resize_height(delta=incr),
                ord("d"): lambda: self._handle_resize_height(delta=-incr),
                ord("f"): lambda: self._handle_resize_width(delta=incr),
                ord("s"): lambda: self._handle_resize_width(delta=-incr),
                # Toggle show/hide class_index
                ord("t"): lambda: self._handle_circle_toggle(),
            }

            # Handle the key using the dispatch dictionary.
            if key in key_handlers:
                key_handlers[key]()

    def _handle_circle_toggle(self):
        """Toggle display of drag corner circle with the 't' key;
        for visual contrast of the class index text."""
        self.show_corner_circle = not self.show_corner_circle

    def _handle_new_box(self):
        """Handle creating a new box with the 'b' key."""
        if self.active_box:
            self.active_box.is_active = False

        display_img_h, display_img_w = self.image_info['h&w']
        self.b_box_counter += 1
        offset = self.b_box_counter * 5
        _x, _y = round(display_img_w * 0.03) + offset, round(display_img_h * 0.03) + offset

        new_box = Box(class_index=self.current_class_index,
                      points=[(_x, _y), (_x + 150, _y + 150)],
                      rotation_angle=0)
        new_box.update_properties()
        new_box.update_points()
        new_box.is_active = True
        self.boxes.append(new_box)
        self.active_box = new_box
        for _box in self.boxes[:-1]:
            _box.is_active = False

    def _handle_clone_box(self):
        """Handle cloning a box with the 'c' key."""
        if not self.active_box:
            return

        img_h, img_w = self.image_info['h&w']
        offset_points: np.ndarray = self.active_box.points.copy()
        right_side, bottom_side = offset_points[2]

        # Need to offset the cloned box down and to the right for visual
        #  clarity and ensure that it does not exceed image boundaries.
        if right_side + self.min_dim >= img_w:
            offset = np.array([-self.min_dim, 0])
        elif bottom_side + self.min_dim >= img_h:
            offset = np.array([0, -self.min_dim])
        else:
            offset = np.array([self.min_dim, self.min_dim])

        offset_points += offset

        cloned_box = Box(class_index=self.active_box.class_index,
                         points=offset_points,
                         rotation_angle=self.active_box.rotation_angle)
        cloned_box.update_properties()
        cloned_box.update_points()
        cloned_box.is_active = True
        self.boxes.append(cloned_box)
        self.active_box = cloned_box
        for _box in self.boxes[:-1]:
            _box.is_active = False

    def _handle_remove_box(self):
        """Handle removing a box with the 'r' key."""
        if self.active_box:
            self.boxes = [box for box in self.boxes if box != self.active_box]
            self.active_box = None

    def _handle_rotate(self, angle_delta):
        """
        Handle rotating a box with arrow keys.
        Args:
            angle_delta: The amount to rotate the box, + or - increment.
        """
        if self.active_box and len(self.active_box.points) == 4:
            self.active_box.rotation_angle += angle_delta
            self.active_box.rotation_angle = max(min(self.active_box.rotation_angle, 180), -180)
            self.active_box.update_points()
            self.check_boundaries(self.active_box)

    def _handle_increase_size(self, delta: int):
        """Handle up arrow key for zoom (Windows) or resize (Linux/Mac)."""
        if MY_OS == 'win':
            self.zoom_level = min(5.0, self.zoom_level + self.zoom_step)
            self.zoom_center = self.last_mouse_pos
        else:  # Linux or macOS
            if self.active_box and len(self.active_box.points) == 4:
                self.active_box.height += delta
                self.active_box.width += delta
                self.active_box.update_points()
                self.check_boundaries(self.active_box)

    def _handle_decrease_size(self, delta: int):
        """Handle down arrow key for zoom (Windows) or resize (Linux/Mac)"""
        if MY_OS == 'win':
            self.zoom_level = max(1.0, self.zoom_level - self.zoom_step)
            self.zoom_center = None if self.zoom_level == 1.0 else self.last_mouse_pos
        else:  # Linux or macOS
            if self.active_box and len(self.active_box.points) == 4:
                self.active_box.height = max(1, self.active_box.height - delta)
                self.active_box.width = max(1, self.active_box.width - delta)
                self.set_min_hw(self.min_dim)
                self.active_box.update_points()

    def _handle_move(self, xy_vector: Union[int, tuple] = (0, 0)):
        """
        Handle moving a box with i, k, j, l keys.
        The i, k, j, and l keys move the box up, down, left, and right.

        Args:
            xy_vector: A tuple representing the x and y pixel movement.
            For example, i = (0, -1), k = (0, 1), j = (-1, 0), l = (1, 0)
        """
        if self.active_box and len(self.active_box.points) == 4:
            self.active_box.center = (self.active_box.center[0] + xy_vector[0],
                                      self.active_box.center[1] + xy_vector[1])
            self.active_box.update_points()
            self.check_boundaries(self.active_box)

    def _handle_resize_height(self, delta: int):
        """
        Handle resizing box height with e, d keys; increase, decrease.
        Args:
            delta: The change in width to apply to the box, + or - increment.
        """
        if self.active_box and len(self.active_box.points) == 4:
            if delta > 0:
                self.active_box.height += delta
                self.check_boundaries(self.active_box)
            else:
                self.active_box.height = max(1, self.active_box.height + delta)
                self.set_min_hw(self.min_dim)

            self.active_box.update_points()

    def _handle_resize_width(self, delta: int):
        """
        Handle resizing box width with f, s keys; increase, decrease.
        Args:
            delta: The change in width to apply to the box, + or - increment.
        """
        if self.active_box and len(self.active_box.points) == 4:
            if delta > 0:
                self.active_box.width += delta
                self.check_boundaries(self.active_box)
            else:
                self.active_box.width = max(1, self.active_box.width + delta)
                self.set_min_hw(self.min_dim)

            self.active_box.update_points()

    def display_to_image_coordinates(self, display_x, display_y):
        """
        Convert display coordinates (where mouse is clicking in zoomed view)
        to original image coordinates; for Windows platform only.

        Args:
            display_x: The x-coordinate in the display window
            display_y: The y-coordinate in the display window

        Returns:
            Tuple of (x, y) coordinates in the original image space
        """

        # If not Windows, or not zoomed, or no zoom center defined,
        #  return original coordinates.
        if MY_OS != 'win' or self.zoom_level == 1.0 or not self.zoom_center:
            return display_x, display_y

        # If zoomed, convert display coordinates to image coordinates
        scale = self.zoom_level
        mouse_x, mouse_y = self.zoom_center

        # Create inverse transformation matrix
        inv_scale = 1.0 / scale
        inv_matrix = np.array([
            [inv_scale, 0, -inv_scale * (1 - scale) * mouse_x],
            [0, inv_scale, -inv_scale * (1 - scale) * mouse_y]
        ], dtype=np.float32)

        # Apply inverse transformation to convert display coordinates to image coordinates
        img_coords = np.array([[display_x, display_y]], dtype=np.float32)
        transformed = cv2.transform(img_coords.reshape(1, 1, 2), inv_matrix)

        return int(transformed[0, 0, 0]), int(transformed[0, 0, 1])

    def handle_mouse_events(self, event, x, y, *args) -> None:
        """
        Handle mouse events for drawing and manipulating boxes.
        The EVENT_ handler expects 5 parameters by default, but only
        three are needed for this application.
        :param event: The mouse event type (e.g., click, move).
        :param x: The x-coordinate of the mouse cursor.
        :param y: The y-coordinate of the mouse cursor.
        :param args: Unused threading.Event arguments for *flags* and *params*.
        """
        # Store mouse position for zoomed image (Windows) function in handle_keys().
        img_x, img_y = self.display_to_image_coordinates(x, y)
        self.last_mouse_pos = (img_x, img_y)

        # Need to stop dragging if the mouse leaves the image area.
        img_h, img_w = self.image_info['h&w']
        if x <= 0 or x >= img_w or y <= 0 or y >= img_h:
            self.is_dragging_box = False
            self.is_dragging_corner = False
            return

        # Define a dispatch dictionary to map events to their handlers.
        event_handlers = {
            cv2.EVENT_LBUTTONDOWN: lambda: self._handle_left_button_down(img_x, img_y),
            cv2.EVENT_RBUTTONDOWN: lambda: self._handle_right_button_down(img_x, img_y),
            cv2.EVENT_RBUTTONUP: lambda: self._handle_right_button_up(img_x, img_y),
            cv2.EVENT_MOUSEMOVE: lambda: self._handle_mouse_move(img_x, img_y),
            cv2.EVENT_LBUTTONUP: lambda: self._handle_left_button_up(img_x, img_y)
        }

        # Handle events using the dispatch dictionary.
        if event in event_handlers:
            event_handlers[event]()

    def _handle_left_button_down(self, x, y):
        """Handle left mouse button press events (selecting and activating boxes)."""
        # Check if the user clicked near the bottom-right corner of the active box.
        if self.active_box and len(self.active_box.points) == 4:
            corner_x, corner_y = self.active_box.points[2]
            if np.hypot(corner_x - x, corner_y - y) < self.get_circle_radius():
                self.is_dragging_corner = True
                return

        # Check if the user clicked inside any box to make it active.
        for _box in reversed(self.boxes):  # self.boxes[::-1]
            if (len(_box.points) == 4 and
                    self.is_point_inside_box(point=(x, y), box_points=_box.points)):
                if self.active_box:
                    self.active_box.is_active = False
                _box.is_active = True
                self.active_box = _box
                self.is_dragging_box = True
                self.offset = (x - _box.center[0], y - _box.center[1])
                return

        # Deactivate the active box if the click is outside all boxes.
        if (self.is_point_outside_all_boxes(point=(x, y), boxes=self.boxes) and
                not self.is_dragging_corner):
            if self.active_box:
                self.active_box.is_active = False
                self.active_box = None

    def _handle_right_button_down(self, x, y):
        """Handle right mouse button press events (starting new box creation)."""

        # Store the start point for the drag operation.
        self.start_point = (x, y)

        # Create a new Box to be drawn
        if self.active_box:
            self.active_box.is_active = False

        new_box = Box(class_index=self.current_class_index)
        new_box.is_active = True
        new_box.points = np.array([
            [x, y],  # Start point
            [x, y],  # Initialize with the same point
            [x, y],  # Will update during drag
            [x, y]  # Will update during drag
        ])
        self.boxes.append(new_box)
        self.active_box = new_box
        self.is_dragging_new_box = True

    def _handle_right_button_up(self, x, y):
        """Handle right mouse button release events (finalizing new box creation)."""
        if hasattr(self, 'is_dragging_new_box') and self.is_dragging_new_box:
            # Finalize the box if it has a minimum size
            if (abs(x - self.start_point[0]) > self.min_dim and
                    abs(y - self.start_point[1]) > self.min_dim):

                # Create box with standard format (top-left, bottom-right points)
                top_left = (min(x, self.start_point[0]), min(y, self.start_point[1]))
                bottom_right = (max(x, self.start_point[0]), max(y, self.start_point[1]))

                self.active_box.points = np.array([
                    top_left,
                    [bottom_right[0], top_left[1]],
                    bottom_right,
                    [top_left[0], bottom_right[1]]
                ])

                self.active_box.update_properties()
                self.active_box.update_points()
                self.check_boundaries(self.active_box)
            else:
                # The new box is too small, remove it
                self.boxes.pop()
                self.active_box = None

            self.is_dragging_new_box = False

    def _handle_mouse_move(self, x, y):
        """Handle mouse movement events (dragging, resizing boxes)."""
        img_h, img_w = self.image_info['h&w']

        # Resize the active box by dragging the bottom-right corner.
        if (self.is_dragging_corner and
                self.active_box and
                len(self.active_box.points) == 4):

            # Need to keep the center fixed while dragging the corner.
            dx = x - self.active_box.center[0]
            dy = y - self.active_box.center[1]

            # Rotate the dragged point back to the unrotated coordinate system.
            angle_rad = np.radians(self.active_box.rotation_angle)

            # Apply the inverse rotation to the dx and dy values to get the
            #  coordinates in the unrotated coordinate system.
            dx_rot = dx * np.cos(-angle_rad) - dy * np.sin(-angle_rad)
            dy_rot = dx * np.sin(-angle_rad) + dy * np.cos(-angle_rad)

            # Update the box width and height, ensuring minimum dimensions.
            cx, cy = self.active_box.center
            new_width = max(2 * abs(dx_rot), self.min_dim)
            new_height = max(2 * abs(dy_rot), self.min_dim)

            # Use new properties when they don't exceed the image boundaries.
            if not (cx - new_width / 2 < 0 or
                    cx + new_width / 2 >= img_w or
                    cy - new_height / 2 < 0 or
                    cy + new_height / 2 >= img_h):
                self.active_box.width = new_width
                self.active_box.height = new_height
                self.active_box.update_points()
                self.check_boundaries(self.active_box)

        # Move the entire active box.
        elif (self.is_dragging_box and
              self.active_box and
              len(self.active_box.points) == 4):
            dx = x - self.offset[0]
            dy = y - self.offset[1]
            self.active_box.center = (dx, dy)
            self.active_box.update_points()
            self.check_boundaries(self.active_box)

        # Update the box during right-button drag to create a new box.
        elif hasattr(self, 'is_dragging_new_box') and self.is_dragging_new_box:
            # Update the box in real-time during drag
            top_left = (min(x, self.start_point[0]), min(y, self.start_point[1]))
            bottom_right = (max(x, self.start_point[0]), max(y, self.start_point[1]))

            self.active_box.points = np.array([
                top_left,
                [bottom_right[0], top_left[1]],
                bottom_right,
                [top_left[0], bottom_right[1]]
            ])

    def _handle_left_button_up(self, x, y):
        """Handle left mouse button release events (ending drag operations)."""
        self.is_dragging_corner = False
        self.is_dragging_box = False
        self.zoom_center = (x, y)  # Used for Windows zooming in handle_keys().

    def check_boundaries(self, box: Optional[Box]) -> None:
        """
        Check if the *box* is within the image boundaries and adjust its
        position if necessary.
        """
        img_h, img_w = self.image_info['h&w']

        # Calculate the bounding box of the rotated rectangle
        min_x, max_x = np.min(box.points[:, 0]), np.max(box.points[:, 0])
        min_y, max_y = np.min(box.points[:, 1]), np.max(box.points[:, 1])

        # Calculate the amount to shift the box to keep it within the image boundaries.
        shift_x = -min_x if min_x < 0 else min(0, img_w - 1 - max_x)
        shift_y = -min_y if min_y < 0 else min(0, img_h - 1 - max_y)

        # Update the center and shift the box points, as needed
        if shift_x or shift_y:
            box.center = (box.center[0] + shift_x, box.center[1] + shift_y)
            box.points += [shift_x, shift_y]
            box.update_points()

    def is_point_inside_box(self, point: tuple, box_points: np.ndarray) -> bool:
        """
        Check if a clicked point is inside a box using the ray casting algorithm.
        Args:
            point: The (x, y) cursor coordinate to check.
            box_points: The four corner points of a box.
        Returns:
            bool: True if the point is inside the box, False otherwise.
        """

        _x, _y = point

        # Need to be able to drag the drag corner of an active box without
        #  moving an underlying box. If there is overlap, then return False
        #  to prevent the inactive underlying box from being moved.
        # The drag corner area is a min_dim square around the bottom-right
        #  corner and should match that used in handle_mouse_events() for
        #  cv2.EVENT_LBUTTONDOWN.
        min_drag_size = self.min_dim // 2
        if self.active_box:
            corner_x, corner_y = self.active_box.points[2]
            if abs(corner_x - _x) <= min_drag_size and abs(corner_y - _y) <= min_drag_size:
                return False

        return cv2.pointPolygonTest(contour=box_points, pt=(_x, _y), measureDist=False) >= 0

    def is_point_outside_all_boxes(self, point: tuple, boxes: list) -> bool:
        """
        Check if a clicked point not in any box.
        Args:
            point: The (x, y) cursor coordinate to check.
            boxes: The list of box objects to check points against.
        Returns:
            bool: True if the point is outside all boxes, False otherwise.
        """

        for _box in boxes:
            if len(_box.points) == 4 and self.is_point_inside_box(point,_box.points):
                return False
        return True

    def put_text_class_index(self,
                             image: np.ndarray,
                             class_idx: str,
                             box_points: np.ndarray,
                             ) -> None:
        """
        Annotate the image with the class index for the box in the bottom-right
        corner of the box
        """

        # Center the size text in the OBB with the 'org' argument.
        #  org: bottom-left corner of the text annotation for an object.
        corner_x, corner_y = box_points[2]
        offset_x, offset_y = Utility.get_text_offsets(
            txt_string=class_idx,
            scale=self.image_info['font scale'],
        )
        text_orig = (round(corner_x - offset_x), round(corner_y + offset_y))

        cv2.putText(img=image,
                    text=class_idx,
                    org=text_orig,
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=self.image_info['font scale'],
                    thickness=2,
                    color=self.cv_color['black'],
                    lineType=cv2.LINE_AA)


class YoloOBBControl(tk.Tk):
    """
    A class to create a Tkinter window for entering the class index used
    for a new OBB box. Sets class indices for OBB object labels.
    Provides a button option to oad a new image file for drawing.
    Also, provides a button option to import existing YOLO object
    detection label data and draw those boxes on the corresponding
    training image. From there the user can adjust size and rotation of
    the boxes and save the results for YOLO OBB model training.
    """
    def __init__(self):
        super().__init__()

        self.entry_label = tk.Label()
        self.class_entry = tk.Entry()
        self.info_txt = tk.StringVar()
        self.info_label = tk.Label()
        self.current_image_name = tk.StringVar()  # Used to display the current image name.
        self.img_name_label = tk.Label()
        self.line_label = tk.Label()
        self.increment_label = tk.Label()
        self.increment = tk.IntVar()  # Used to set the size and rotation step factor.

        # Want the 'Get new image' button bg to match the image name label fg
        #  when focusOut. When focusIn, image name label fg uses a better contrast.
        #  The color values are X11 color names recognized by tkinter.
        self.color = {
            'img label': 'gold',
            'window': 'gray75',
            'dark': 'gray30',
            'save button': 'dodger blue',
            'increment': 'dark orange',
            'black': 'gray0',
        }

    def set_color_focusout(self):
        self.config(bg=self.color['dark'])
        self.entry_label.configure(bg=self.color['dark'])
        self.info_label.config(bg=self.color['dark'], fg=self.color['dark'],)
        self.img_name_label.config(bg=self.color['dark'], fg=self.color['img label'])
        self.line_label.config(bg=self.color['dark'], fg=self.color['increment'])
        self.increment_label.config(bg=self.color['dark'], fg=self.color['increment'])

    def set_color_focusin(self):
        self.config(bg=self.color['window'],)
        self.entry_label.configure(bg=self.color['window'],)
        self.info_label.config(bg=self.color['window'], fg=self.color['black'],)
        self.img_name_label.config(bg=self.color['window'], fg=self.color['save button'])
        self.line_label.config(bg=self.color['window'], fg='black')
        self.increment_label.config(bg=self.color['window'], fg='black')

    def config_control_window(self):
        self.title('YOLO OBB Control')
        self.attributes("-topmost", True)
        self.geometry('390x300+100+400')  # width x height + x_offset + y_offset
        self.config(borderwidth=6, relief='groove')
        self.protocol('WM_DELETE_WINDOW', self.on_close)

        self.grid_columnconfigure(0,weight=1)
        self.grid_rowconfigure(list(range(9)), weight=1)  # Assuming 8 rows

        self.entry_label.config(text='Enter a YOLO class index number for the next box:')
        self.current_image_name.set(f'Current image: {Path(box_drawer.image_path).stem}')
        self.img_name_label.config(textvariable=self.current_image_name,)
        self.line_label.config(text='Line thickness', )
        self.increment_label.config(text='Increment')
        self.increment.set(1)  # Default value for incrementing px size and rotation angle.

        # This text will be replaced with save metrics when the user saves.
        self.info_txt.set(
            f"A YOLO labels file in the {LABELS_FOLDER} folder\n"
            'can draw those boxes on its corresponding image.')
        self.info_label.config(
            textvariable=self.info_txt,
            width=65,  # character width is a function of font size and platform
            font='TkTooltipFont',  # TkSmallCaptionFont
        )

        self.class_entry.config(width=3)
        self.class_entry.insert(index=tk.INSERT, string='0')  # Default value for class index.

        self.bind('<Escape>', lambda _: self.on_close())
        self.bind('<Control-q>', lambda _: self.on_close())
        self.bind('<FocusIn>', lambda _: self.set_color_focusin())
        self.bind('<FocusOut>', lambda _: self.set_color_focusout())
        self.class_entry.bind('<Return>', lambda _: self.set_class_index())
        self.class_entry.bind('<FocusOut>', lambda _: self.set_class_index())

        # Create buttons and radio buttons for the control window.
        # '\uFF0B' Full-width plus sign, '\uFF0D' Full-width minus sign.
        buttons_and_rows = (
            (tk.Button(text='Load new image file', command=box_drawer.open_image,
                       width=15, background=self.color['img label']), 3),
            (tk.Button(text='Save to results', command=self.on_save,
                       width=15, background=self.color['save button']), 5),
            (tk.Button(text='＋', command=Utility.increase_line_thickness,
                       width=1, background=self.color['increment']), 6),
            (tk.Button(text='－', command=Utility.decrease_line_thickness,
                       width=1, background=self.color['increment']), 7),
            (tk.Button(text='Help', command=Utility.show_help,
                       background=self.color['window']), 8),
        )

        # Need three radio buttons for incrementing px size and rotation angle.
        radiobuttons_and_rows = [
            (tk.Radiobutton(text=str(i), value=i, variable=self.increment,
                            background=self.color['increment']), 6)
            for i in range(1, 4)
        ]

        # Place widgets in the grid. (Centered is window_w//2 - text_w//2)
        self.entry_label.grid(row=0, column=0, padx=10, pady=(5, 0), sticky=tk.EW)
        self.class_entry.grid(row=1, column=0, pady=5)
        self.img_name_label.grid(row=2, column=0, padx=10, pady=0, sticky=tk.EW)
        self.info_label.grid(row=4, column=0, padx=10, pady=(0, 10), sticky=tk.EW)
        self.increment_label.grid(row=7, column=0,
                                  padx=(0, 165), # centered
                                  pady=(0, 10), sticky=tk.E)
        self.line_label.grid(row=8, column=0, padx=(0, 10), pady=(0, 10), sticky=tk.E)

        for button, row in buttons_and_rows:
            button.grid(
                row=row, column=0, padx=30, pady=5 if row != 6 else (10, 0),
                sticky=tk.EW if row < 6 else tk.E)
            if row == 8:  # Help button, bottom left corner
                button.grid_configure(padx=10, sticky=tk.W)

        for i, (radiobutton, row) in enumerate(radiobuttons_and_rows):
            radiobutton.grid(
                row=row, column=0, padx=(0, 225 - i * 50), pady=(10, 0), sticky=tk.E)


    def set_class_index(self):
        """
        Sets Box.class_index for the current box. This is the index of
        the class label for the box, as required for YOLO OBB label data.
        Called from class_entry.bind() methods.
        """

        # Replace display of invalid class index in Entry field when
        #  Control window looses focus.
        try:
            box_drawer.current_class_index = abs(int(self.class_entry.get()))
        except ValueError:
            messagebox.showerror(title='Invalid class index',
                                 detail='Please enter an integer value.')
            self.class_entry.delete(first=0, last=tk.END)  # Clear the entry field.
            self.class_entry.insert(index=tk.INSERT, string='0')  # Reset default value.

    def get_labels(self, labels_path: str) -> Union[list, None]:
        """
        Generates label data to be used by the view_labels() method.
        Called from BoxDrawer.look_for_labels().

        Args:
             labels_path: (string) The path to the YOLO labels file to load.
        Returns: A list of labels to convert.
        """

        self.current_image_name.set(
            f"Current image: {box_drawer.image_info['full name']}")

        try:
            with open(labels_path, 'r') as labels:
                labels_to_convert = labels.readlines()
        except FileNotFoundError:
            messagebox.showerror(title='File not found', message=f'{labels_path} does not exist.')
            return None

        if not labels_to_convert:
            messagebox.showerror(
                title='Empty file',
                message=f'{Path(labels_path).name} is empty.',
                detail='Please select a valid YOLO labels file.'
            )
            return None

        for line, label in enumerate(labels_to_convert):
            if not label.split()[0].isdigit():
                messagebox.showerror(
                    title='Bad class index',
                    detail='The class index must be an integer.\n'
                           f'Check line: {line +1}. No boxes were drawn.\n'
                )
                return None

        # Now check the first line for expected number of data elements.
        if not len(labels_to_convert[0].split()) in (5, 9):
            messagebox.showerror(
                title='Bad input format',
                detail=f'{Path(labels_path).name} cannot be used.\n'
                       'Expect either 5 or 9 data elements per line.\n'
                       '  - YOLO: class_index center-x center-y w h\n'
                       '  - YOLO-OBB: class_index x1 y1 x2 y2 x3 y3 x4 y4\n'
                       'Only label data; no comments, no headers.'
            )
            return None

        return labels_to_convert

    def on_save(self):
        """
        Save to a text file boxes class index and point coordinates
        in YOLO OBB format. Also save the drawn image as a jpeg file.
        Files are saved to the local 'results' folder.
        Called from the 'Save' button in config_control_window()
        """
        _boxes = box_drawer.boxes
        _img_info = box_drawer.image_info

        # Need to remove any non-4-point elements for an accurate box count.
        #  This may not be necessary, but it is a good practice.
        _boxes = [_box for _box in _boxes if len(_box.points) == 4]

        if not _boxes:
            self.info_txt.set('No boxes to save.')
            return

        img_name = _img_info['short name']

        # Create the results directory if it doesn't exist.
        Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

        # Box.points are in absolute coordinates. So, need to convert pixels to
        #  normalized coordinates (0.0 to 1.0).
        # Write the data to a text file as yolo-obb labels and draw boxes on the image.
        with open(f'{RESULTS_DIR}/{img_name}.txt', 'w') as result_file:
            for _box in _boxes:
                if len(_box.points) == 4:
                    obb_label = Utility.convert_to_obb_label_format(
                        box=_box, im_size=_img_info['h&w'])
                    result_file.write(f'{obb_label}\n')

        cv2.imwrite(f"{RESULTS_DIR}/{img_name}_result.jpg",
                    box_drawer.display_image)

        app.info_txt.set(f'{len(_boxes)} YOLO OBB labels, and the annotated image,\n'
                         ' were saved to the results folder.')

        # Need to provide a session record of save actions for the user.
        print(f'{len(_boxes)} YOLO OBB labels and annotated image for {img_name}'
              f' were saved to folder "{RESULTS_DIR}".')

    def on_close(self):
        """
        Allows clean closing or Tkinter and cv2 windows. Exit by using
        main window 'X' button or Esc key press.
        Called from YoloInfoWindow __init__() window protocol and button.
        """

        # Stop the while loop, causing draw_box() to destroy the CV
        #  window, then destroy the Tk app and its draw_box() thread.
        #  sys.exit() is just in case.
        box_drawer.stop_loop.set()
        self.destroy()
        print('User quit the program.')
        sys.exit(0)


class Utility:
    """
    A class of static methods to provide utility functions.
    Methods include:
        view_labels
        convert_from_obb_label_format
        convert_to_obb_label_format
        increase_line_thickness
        decrease_line_thickness
        get_text_offsets
        show_help
    """

    @staticmethod
    def view_labels(label_data: list):
        """
        Provides Box formatting and BoxDrawer display of YOLO object
        detection data. Calls get_labels() to get the data
        to convert. Then converts the normalized coordinates to absolute
        pixel coordinates and creates a new Box object for each set of
        coordinates. Finally, appends the new Box objects to the boxes
        list and updates the display image with the new boxes.
        Called from the 'Convert label data' button in the Tkinter window
        and from BoxDrawer.look_for_labels().

        Args:
            label_data (tuple): YOLO OBB label, image height, image width.
            Argument as passed from BoxDrawer.look_for_labels().
        """

        # Clear existing boxes before adding new ones.
        box_drawer.boxes.clear()
        labels_to_view = label_data
        img_h, img_w = box_drawer.image_info['h&w']

        # Check format of the first line of the labels file to determine
        #  if the data is standard YOLO bounding boxes or OBB boxes
        for line, label in enumerate(labels_to_view):
            if label.strip():
                try:
                    data = list(map(float, label.split()))
                    if len(data) == 5:  # Standard YOLO format
                        class_index, center_x, center_y, _w, _h = data
                        x_abs, y_abs = int(center_x * img_w), int(center_y * img_h)
                        w_abs, h_abs = int(_w * img_w), int(_h * img_h)
                        points = [(x_abs - w_abs // 2, y_abs - h_abs // 2),
                                  (x_abs + w_abs // 2, y_abs + h_abs // 2)]
                        angle = 0
                    elif len(data) == 9:  # OBB format
                        class_index, *points = data
                        points, angle = Utility.convert_from_obb_label_format(
                            pts=points, im_size=(img_h, img_w))
                    else:
                        raise ValueError

                    new_box = Box(class_index=int(class_index),
                                  points=points,
                                  rotation_angle=angle)
                    new_box.is_active = False
                    box_drawer.boxes.append(new_box)
                    box_drawer.active_box = new_box
                    new_box.update_properties()
                    new_box.update_points()
                    box_drawer.check_boundaries(new_box)

                except ValueError:
                    messagebox.showerror(
                        title='Invalid data',
                        detail=f'Data in line: {line + 1} cannot be used.\n'
                               'Not all boxes were drawn.\n'
                               'Please check the data and try again.'
                    )
                    return

        if box_drawer.boxes:
            messagebox.showinfo(
                title='Conversion complete',
                detail=f'{len(box_drawer.boxes)} OBB boxes were created'
                       ' from the YOLO labels file.'
            )

    @staticmethod
    def convert_from_obb_label_format(pts: list,
                                      im_size: tuple) -> tuple:
        """
        Convert YOLO oriented bounding box (OBB) labels to a format
        suitable for viewing and further processing. This function
        converts normalized OBB points to absolute pixel coordinates.
        Called from view_labels().

        Args:
            pts (list): Normalized coordinates for the four corners of the OBB.
            im_size (tuple): Image pixel height and width (img_h, img_w).
        Returns:
             tuple (list, float):
            - A list of tuples representing the absolute pixel coordinates of the
              bounding box corners in the format [(x1, y1), (x2, y2)].
            - The rotation angle in degrees for the OBB.
        """

        x1, y1, x2, y2, x3, y3, x4, y4 = pts

        img_h, img_w = im_size

        # Calculate the center point, width, and height of the OBB, as
        #  relative float values.
        center_x = (x1 + x2 + x3 + x4) / 4
        center_y = (y1 + y2 + y3 + y4) / 4
        width = np.hypot(x2 - x1, y2 - y1)
        height = np.hypot(x3 - x2, y3 - y2)

        # Calculate the angle of rotation, theta. May be negative, so use atan2.
        # This will give the angle in radians, which is converted to degrees for
        # compatibility with cv2.getRotationMatrix2D in Box.update_properties().
        theta = np.degrees(np.arctan2(y2 - y1, x2 - x1))

        # Convert normalized coordinates to absolute pixel coordinates for display.
        # Need to use round() for best accuracy in pixel coordinates.
        x_abs, y_abs = round(center_x * img_w), round(center_y * img_h)
        w_abs, h_abs = round(width * img_w), round(height * img_h)

        # Return format: [(x1, y1), (x2, y2)] and rotation, for a Box object.
        return [(round(x_abs - w_abs * 0.5), round(y_abs - h_abs * 0.5)),
                (round(x_abs + w_abs * 0.5), round(y_abs + h_abs * 0.5))], theta

    @staticmethod
    def convert_to_obb_label_format(box: Box, im_size: tuple) -> str:
        """
        Correctly converts pixel coordinates to normalized YOLO OBB format while
        preserving each box's exact geometry (aspect ratio, rotation) by normalizing
        each dimension separately relative to image size.
        Called from on_save(), passing one box at a time from a loop.

        Args:
            box: The Box object with pixel coordinates.
            im_size: (image px height, image px width)

        Returns: (string) The formatted YOLO OBB label string for the *box*;
            class_index, x1_norm, y1_norm, ..., x4_norm, y4_norm
        """
        img_h, img_w = im_size
        points = box.points

        # Calculate center in pixels
        center_px = np.mean(points, axis=0)

        # Calculate width and height in pixels
        width_px = np.linalg.norm(points[1] - points[0])
        height_px = np.linalg.norm(points[3] - points[0])

        # Convert center, width, and height to normalized coordinates
        center_x_norm, center_y_norm = center_px / [img_w, img_h]
        w_norm, h_norm = width_px / img_w, height_px / img_h

        # Corners relative to center before rotation
        half_w, half_h = w_norm / 2, h_norm / 2
        corners = np.array([
            [-half_w, -half_h],
            [ half_w, -half_h],
            [ half_w,  half_h],
            [-half_w,  half_h]
        ])

        # Calculate rotation angle in radians
        angle_rad = np.arctan2(
            points[1][1] - points[0][1],  # y2 - y1 vector
            points[1][0] - points[0][0]  # x2 - x1 vector
        )

        # Make and apply the rotation matrix shift.
        rot = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad),  np.cos(angle_rad)]
        ])
        rotated_corners = corners @ rot.T

        normalized_points = rotated_corners + [center_x_norm, center_y_norm]

        # Convert numpy array of points to formatted strings, then return the label.
        label_points = [f"{point:.6f}" for point in normalized_points.flatten()]

        return str(box.class_index) + ' ' + ' '.join(label_points)

    @staticmethod
    def increase_line_thickness():
        """
        Increase the line thickness of the drawn boxes and font.
        """
        box_drawer.image_info['line thickness'] += 1

    @staticmethod
    def decrease_line_thickness():
        """
        Decrease the line thickness of the drawn boxes and font.
        """
        box_drawer.image_info['line thickness'] = max(1, box_drawer.image_info['line thickness'] - 1)

    @staticmethod
    def get_text_offsets(txt_string: str,
                         scale: float) -> tuple[float, int]:
        """
        Calculate the x and y position correction factors to help center
        *txt_string* in cv2.putText() for annotating objects.
        Called from annotate_object().

        Args:
            txt_string: A string of the object's size to display.
            scale: (float) The font scale of the text_string.
        Returns:
            A tuple of x and y position adjustment factors for text centering.
        """

        (txt_width, _), baseline = cv2.getTextSize(
            text=txt_string,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=scale,
            thickness=1,
        )
        offset_x = txt_width / 2

        return offset_x, baseline

    @staticmethod
    def show_help():
        """
        Show help documentation from the program docstring in a pop-up
        window. Called from cv window with 'h' key.
        Returns: None
        """
        help_win = tk.Toplevel()
        help_win.title('Info and help tips')
        help_win.wm_minsize(595, 200)  # May be platform dependent.
        help_text = ScrolledText(master=help_win,
                                 width=72, # 72 works for Linux and Windows.
                                 height=25,
                                 bg='dark slate grey',
                                 fg='grey95',
                                 relief='groove',
                                 borderwidth=8,
                                 padx=30, pady=30,
                                 wrap=tk.WORD,
                                 )
        help_text.insert(tk.INSERT, __doc__)
        help_text.pack(fill='both', side='left', expand=True)


def main():
    """
    Main function to run the program, with threaded CV functions.
    """

    app.config_control_window()

    # Run update_image_info() after app Tk window is initialized because
    #  it uses Tk winfo_screenwidth() and winfo_screenheight().
    box_drawer.update_image_info()

    # Run live updates of threaded cv2 events with the while loop in draw_box().
    threading.Thread(target=box_drawer.draw_box).start()

    print(f'{Path(sys.modules["__main__"].__file__).stem} now running...')
    app.mainloop()


if __name__ == "__main__":

    # Instantiate the drawing class with the default image.
    # Loading a starting image is necessary for current flow architecture.
    box_drawer = BoxDrawer(image_path='images/start_image.jpg')

    # Instantiate the Tkinter control window; tk.Tk is inherited.
    # Call main() to start the application.
    app = YoloOBBControl()
    main()
