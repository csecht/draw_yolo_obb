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
Option 2: Click on the image, press 'n', then right-click two points to
define the top-left and bottom-right corners of the box.
Option 3: Click on an existing box, press 'c' to clone it, then drag the
new box to a new location.

To move and reshape a box: click inside it to activate, then drag to
reposition. Click and drag the dot in the bottom-right corner to resize.
Or use the keyboard, as described below. The active box is highlighted
in red, inactive boxes in green. Only an active box can be manipulated
with the mouse or keyboard.

Use the YOLO OBB Control window to enter a class index for the box to
be drawn. The default class index is 0. Enter the index for a new class
index BEFORE clicking the image and pressing 'n' or 'b'. The number
entered will be used for all subsequent boxes until changes. Class index
numbers are displayed in the box's corner with the drag dot.

Keyboard manipulation of the active box:
The left and right arrow keys rotate the box in 3 degree increments.
The up and down arrow keys increase or decrease the box size in Linux.
The 'i', 'k', 'j', and 'l' keys move the box up, down, left, and right.
The 'e', 'd', 'f', and 's' keys increase or decrease height and width.
The 'r' key removes the active (red) box.

Program control keys:
The up and down arrow keys zoom the image in and out in Windows.
The mouse wheel zooms in and out in Linux.
The 'Esc' key quits program from the control window.
The 'X' button in the control window bar quits the program.
The 'h' key, from the image window, pops up a scrolling help window.

If you are rt-clicking to draw a box and nothing is happening, remember
to first press the 'n' key, then right-click on two positions.

The terminal window provides feedback on save actions and errors. It may
be covered by the main image window that fills most of the screen, so
reposition windows as needed.

Clicking the image window's 'X' button will not end the program, it just
redraws the window. Quit the program from the YOLO OBB Control window
with Esc or its window close button, 'X'.
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

        if len(self.points) == 2:
            # Axially oriented box: calculate center, width, and height.
            self.center = tuple(np.mean(self.points, axis=0).astype(int))
            self.width, self.height = np.abs(self.points[1] - self.points[0])

        elif len(self.points) == 4:
            # Rotated box: calculate center, width, and height.
            # Rotate points to align with axes for width and height calculation.
            # The `None` indexing adds a third dimension to the array for
            # `cv2.transform`. The `[0]` indexing extracts the transformed
            # points to re-form a 2D array with new corner coordinates.
            self.center = tuple(np.mean(self.points, axis=0))
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

        self.window_name = ''

        self.image_array: np.ndarray = cv2.imread(image_path)
        self.image_path: str = image_path  # The default start image
        self.image_name: str = Path(image_path).stem  # File name without extension
        self.image_info = {}  # Dictionary populated in update_image_info().

        self.labels_folder = 'labels'  # Default local folder for yolo labels files.

        # Note that list definition here won't pick up files added during program run.
        self.labels_files = [f.name for f in Path(self.labels_folder).glob('*.txt')]

        self.boxes = []  # List to store all box coordinates
        self.active_box = None  # Currently active box
        self.is_dragging_corner = False  # Tracks if a corner is being dragged
        self.is_dragging_box = False  # Tracks if the entire box is being dragged
        self.offset = (0, 0)  # Offset for dragging the entire box
        self.b_box_counter = 0  # Counter for 'b' key presses
        self.current_class_index = 0

        # Minimum dimension pixel size to prevent disappearing, too small,
        #  and out-of-bounds objects,
        self.min_dim = 12

        # cv2 colors for drawing boxes and text.
        self.cv_font_color = {
            'white': (255, 255, 255),
            'black': (0, 0, 0),  # Black color for class index text
            'red': (0, 0, 255),  # Red color for active box
            'green': (0, 255, 0),  # Green color for boxes
            'cyan': (255, 255, 0),  # Cyan color for dot to drag box corner
            'orange': (0, 159, 230),  # Orange color for saved class index circle
            'blue': (255, 0, 0),
        }

        self.zoom_level = 1.0
        self.zoom_step = 0.1  # Same as Linux default
        self.zoom_center = None  # Mouse position when zooming
        self.last_mouse_pos = (0, 0)  # Track mouse position
        self.zoom_key_active = False  # Track if zoom modifier key is pressed

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
        Path(self.labels_folder).mkdir(parents=True, exist_ok=True)

        labels_file = Path(img_path).stem + '.txt'
        with self.control_lock:
            if labels_file in self.labels_files:
                convert_now = messagebox.askyesno(
                    parent=app,
                    title='Convert label data?',
                    detail=f'The image "{Path(img_path).name}"'
                           f' has a labels file in {self.labels_folder}.\n'
                           f'"Yes" draws those boxes on the image.\n')
                if convert_now:
                    # Load the existing label data and draw boxes on the image.
                    labels_path = f'{self.labels_folder}/{labels_file}'
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
        image_copy: ndarray = self.image_array.copy()
        img_h, img_w, _ = self.image_array.shape

        # Need the maximum CV window to be smaller than the screen size,
        #  and comfortably scale-to-fit the largest image dimension.
        win_h = app.winfo_screenheight() * 0.85
        win_w = app.winfo_screenwidth() * 0.85
        display_factor = max(img_h / win_h, img_w / win_w) if img_h > win_h or img_w > win_w else 1

        # Ideas for scaling: https://stackoverflow.com/questions/52846474/
        #   how-to-resize-text-for-cv2-puttext-according-to-the-image-object_size-in-opencv-python
        # The conversion factors were empirically determined.
        avg_d = (win_h + win_w) * 0.5
        line_thickness = max(round(avg_d * 0.001 * display_factor), 2)
        font_scale = max(avg_d * 0.0002, 0.4) * display_factor

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
        drawing.
        """

        # CV WINDOW SETUP
        self.window_name = "View and Edit OBB ('h' for help)"
        cv2.namedWindow(self.window_name,cv2.WINDOW_GUI_NORMAL,)
        cv2.resizeWindow(
            self.window_name,
            self.image_info['window_w'],
            self.image_info['window_h'])

        cv2.setMouseCallback(self.window_name,
                             self.handle_mouse_events)

        # BOX DRAWING and KEYBOARD ACTION LOOP
        # The loop provides live drawing updates as long as the threading.Event
        #  is not set. Event is set upon call to on_close().
        while not self.stop_loop.is_set():

            # Using a copy allows live drawing of boxes.
            display_image = self.image_info['copy'].copy()

            if MY_OS == 'win' and self.zoom_level != 1.0 and self.zoom_center:
                # Zooms window the up/down arrow keys on Windows.
                _h, _w = display_image.shape[:2]
                img_h, img_w = self.image_info['h&w']
                scale = self.zoom_level

                # Convert mouse position to image coordinates
                mouse_x, mouse_y = (int(coord * scale) for coord, scale in
                                    zip(self.zoom_center, (_w / img_w, _h / img_h)))

                # Create transformation matrix centered on the mouse position.
                t_matrix = np.array([
                    [scale, 0, (1 - scale) * mouse_x],
                    [0, scale, (1 - scale) * mouse_y]
                ])  #, dtype=np.float32)

                # Apply transformation
                display_image = cv2.warpAffine(
                    display_image,
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
                    color = (self.cv_font_color['green']
                             if not _box.is_active else self.cv_font_color['red'])

                    cv2.polylines(img=display_image,
                                  pts=[pts],
                                  isClosed=True,
                                  color=color,
                                  thickness=self.image_info['line thickness'],
                                  )

                    # Draw a circle in the bottom-right corner of the active box.
                    # Make radius size relative to image size.
                    if _box.is_active:
                        cv2.circle(display_image,
                                   center=pts[2],
                                   radius=self.get_circle_radius(),
                                   color=self.cv_font_color['cyan'],
                                   thickness=cv2.FILLED,
                                   )

                    self.put_text_class_index(image=display_image,
                                              class_idx=str(_box.class_index),
                                              box_points=pts,
                                              )

            cv2.imshow(self.window_name, display_image)

            self.set_keys()

        cv2.destroyWindow(self.window_name)

    def set_keys(self) -> None:
        """
        Define all the key events for manipulating boxes in the CV window.
        Called from draw_box() while the image is displayed from loop.
        """
        if MY_OS == 'lin':
            key = cv2.waitKey(1)  # restricts keycodes to 0-255
            left_arrow = 81
            right_arrow = 83
            up_arrow = 82
            down_arrow = 84
        else:  # 'win'
            key = cv2.waitKeyEx(1)  # for Windows special keys (on HP Pavilion)
            left_arrow = 2424832  # VK_LEFT, 0x25
            right_arrow = 2555904 # VK_RIGHT, 0x27
            up_arrow = 2490368  # VK_UP, 0x26
            down_arrow = 2621440  # VK_DOWN, 0x28

        # Need image height and width for positioning boxes with the 'b' key.
        display_img_h, display_img_w = self.image_info['h&w']

        # DEV: if changed, needs to be a factor of 360 rotation.
        angle_increment = 1

        # Note: key functions require the cv2 window to be in focus (click image).
        # Note: current_class_index is set in YoloOBBControl.set_class_index().

        # Press 'n' to start a new box when placing a box with two right button
        #  clicks. The two coordinates are used to define the top-left and
        #  bottom-right corners of the box, which are processed in
        #  handle_mouse_events() for the new boxes.points array.
        if key == ord("n"):
            self.active_box = None
            self.boxes.append(Box(class_index=self.current_class_index))
            self.boxes[-1].is_active = True
            for _box in self.boxes[:-1]:
                _box.is_active = False

        # Auto-draw a new box near the top-left corner of the image.
        #  Pressing 'b' multiple times will overlay each new box.
        #  Therefore, need to offset each new box so that user can
        #  see that there are multiple boxes.
        if key == ord("b"):
            if self.active_box:
                self.active_box.is_active = False

            # Increment box counter and calculate offset for new box
            self.b_box_counter += 1
            offset = self.b_box_counter * 5
            x, y = round(display_img_w * 0.03) + offset, round(display_img_h * 0.03) + offset

            # Create and activate the new box
            new_box = Box(class_index=self.current_class_index,
                          points=[(x, y), (x + 150, y + 150)],
                          rotation_angle=0)
            new_box.update_properties()
            new_box.update_points()
            new_box.is_active = True
            self.boxes.append(new_box)
            self.active_box = new_box

            # Deactivate all other boxes
            for box in self.boxes[:-1]:
                box.is_active = False

        # Press 'c' to clone (copy and paste) the active box.
        if key == ord("c"):
            if self.active_box:  # len(self.active_box.points) == 4

                # Need to offset the cloned points to avoid overlap, and
                #  move away from a nearby image edge.
                offset_points: np.ndarray = self.active_box.points.copy()
                right_side, bottom_side = offset_points[2]

                # Determine the edge offset based on the position of the box.
                if right_side + self.min_dim >= display_img_w:
                    offset = np.array([-self.min_dim, 0])
                elif bottom_side + self.min_dim >= display_img_h:
                    offset = np.array([0, -self.min_dim])
                else:
                    offset = np.array([self.min_dim, self.min_dim])

                # Apply the offset to all points.
                offset_points += offset

                # Note that here four points are used, not two as
                #  with the 'b' or 'n' keys. Rotated points will be
                #  transformed in the Box class methods to preserve box
                #  properties.
                cloned_box = Box(class_index=self.active_box.class_index,
                                 points=offset_points,
                                 rotation_angle=self.active_box.rotation_angle)
                self.boxes.append(cloned_box)
                cloned_box.is_active = True
                self.active_box = cloned_box
                for _box in self.boxes[:-1]:
                    _box.is_active = False
                cloned_box.update_properties()
                cloned_box.update_points()

        # Press 'r' to remove the active box.
        if key == ord("r"):
            if self.active_box:
                self.boxes = [box for box in self.boxes if box != self.active_box]
                self.active_box = None

        # Rotate active box left (counter-clockwise), apply limit.
        if key == left_arrow:
            if self.active_box and len(self.active_box.points) == 4:
                self.active_box.rotation_angle -= angle_increment
                self.active_box.rotation_angle = max(self.active_box.rotation_angle, -180)
                self.active_box.update_points()
                self.check_boundaries(self.active_box)

        # Rotate active box right (clockwise), apply limit.
        if key == right_arrow:
            if self.active_box and len(self.active_box.points) == 4:
                self.active_box.rotation_angle += angle_increment
                self.active_box.rotation_angle = min(self.active_box.rotation_angle, 180)
                self.active_box.update_points()
                self.check_boundaries(self.active_box)

        if MY_OS == 'lin':
            # Simultaneously increase height and width.
            if key == up_arrow:
                if self.active_box and len(self.active_box.points) == 4:
                    self.active_box.height += 1
                    self.active_box.width += 1
                    self.active_box.update_points()
                    self.check_boundaries(self.active_box)

            # Simultaneously decrease height and width.
            if key == down_arrow:
                if self.active_box and len(self.active_box.points) == 4:
                    self.active_box.height = max(1, self.active_box.height - 1)
                    self.active_box.width = max(1, self.active_box.width - 1)
                    self.set_min_hw(self.min_dim)
                    self.active_box.update_points()

        else:  # is Windows: Handle zoom in/out.
            if key == up_arrow:  # Zoom in
                self.zoom_level = min(5.0, self.zoom_level + self.zoom_step)

                # Need to keep zooming active after return to no-zoom.
                self.zoom_center = self.last_mouse_pos

            elif key == down_arrow:  # Zoom out
                # Reset when fully zoomed out.
                self.zoom_level = max(1.0, self.zoom_level - self.zoom_step)
                self.zoom_center = None if self.zoom_level == 1.0 else self.last_mouse_pos


        # The following layout of keys for manipulating the active box
        #  is based on the QWERTY keyboard layout.
        #  The rationale is that keys' relative positions infer their
        #  action, similar to the spatial layout of the
        #  arrow keys cluster.

        # Move active box up
        if key == ord("i"):
            if self.active_box and len(self.active_box.points) == 4:
                self.active_box.center = (self.active_box.center[0],
                                          self.active_box.center[1] - 1)
                self.active_box.update_points()
                self.check_boundaries(self.active_box)

        # Move active box down
        if key == ord("k"):
            if self.active_box and len(self.active_box.points) == 4:
                self.active_box.center = (self.active_box.center[0],
                                          self.active_box.center[1] + 1)
                self.active_box.update_points()
                self.check_boundaries(self.active_box)

        # Move active box left
        if key == ord("j"):
            if self.active_box and len(self.active_box.points) == 4:
                self.active_box.center = (self.active_box.center[0] - 1,
                                          self.active_box.center[1])
                self.active_box.update_points()
                self.check_boundaries(self.active_box)

        # Move active box right.
        if key == ord("l"):
            if self.active_box and len(self.active_box.points) == 4:
                self.active_box.center = (self.active_box.center[0] + 1,
                                          self.active_box.center[1])
                self.active_box.update_points()
                self.check_boundaries(self.active_box)

        # Increase the height of the active box.
        if key == ord("e"):
            if self.active_box and len(self.active_box.points) == 4:
                self.active_box.height += 1
                self.active_box.update_points()
                self.check_boundaries(self.active_box)

        # Decrease the height of the active box.
        if key == ord("d"):
            if self.active_box and len(self.active_box.points) == 4:
                self.active_box.height = max(1, self.active_box.height - 1)
                self.set_min_hw(self.min_dim)
                self.active_box.update_points()

        # Increase the width of the active box.
        if key == ord("f"):
            if self.active_box and len(self.active_box.points) == 4:
                self.active_box.width += 1
                self.active_box.update_points()
                self.check_boundaries(self.active_box)

        # Decrease the width of the active box
        if key == ord("s"):
            if self.active_box and len(self.active_box.points) == 4:
                self.active_box.width = max(1, self.active_box.width - 1)
                self.set_min_hw(self.min_dim)
                self.active_box.update_points()

        # Press 'h' to display help documentation in a pop-up window.
        if key == ord("h"):
            # print(__doc__)
            Utility.show_help()

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

        # Need to restrict mouse events to the image boundary.
        img_h, img_w = self.image_info['h&w']

        # Store mouse position for zoomed image (Windows) function in set_keys().
        # Moving self.zoom_center = (x, y) from the cv2.EVENT_LBUTTONUP event to
        #  here would allow panning with mouse movement.
        self.last_mouse_pos = (x, y)

        # Need to stop dragging if the mouse leaves the image area.
        if x <= 0 or x >= img_w or y <= 0 or y >= img_h:
            self.is_dragging_box = False
            self.is_dragging_corner = False
            return

        if event == cv2.EVENT_LBUTTONDOWN:

            # Check if the user clicked near the bottom-right corner of the active box.
            if self.active_box and len(self.active_box.points) == 4:
                corner_x, corner_y = self.active_box.points[2]
                if np.hypot(corner_x - x, corner_y - y) < self.get_circle_radius():
                    self.is_dragging_corner = True

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
                    break

            # Deactivate the active box if the click is outside all boxes.
            if (self.is_point_outside_all_boxes(point=(x, y), boxes=self.boxes) and
                 not self.is_dragging_corner):
                if self.active_box:
                    self.active_box.is_active = False
                self.active_box = None

        elif event == cv2.EVENT_RBUTTONDOWN:

            # If no box is active, start drawing a new box following 'r'
            #  keypress. Two rt-clicks set the top-left and bottom-right
            #  corners of a new box.
            # New box starts with default points array, [[0, 0], [0, 0]].
            # Check if we have a newly created box with zero coordinates.
            if (not self.active_box and len(self.boxes) > 0 and
                    np.any(np.sum(self.boxes[-1].points, axis=1) == 0)):

                # Find the first zero point and set it to the click position
                zero_indices = np.where(np.sum(self.boxes[-1].points, axis=1) == 0)[0]
                if zero_indices.size > 0:
                    self.boxes[-1].points[zero_indices[0]] = np.array([x, y])

                # If all points are now non-zero, finalize the box
                if np.all(np.sum(self.boxes[-1].points, axis=1) > 0):
                    self.boxes[-1].update_properties()
                    self.boxes[-1].update_points()
                    self.boxes[-1].is_active = True
                    self.active_box = self.boxes[-1]

        elif event == cv2.EVENT_MOUSEMOVE:

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

            elif (self.is_dragging_box and
                  self.active_box and
                  len(self.active_box.points) == 4):
                # Move the entire active box.
                dx = x - self.offset[0]
                dy = y - self.offset[1]
                self.active_box.center = (dx, dy)
                self.active_box.update_points()
                self.check_boundaries(self.active_box)

        elif event == cv2.EVENT_LBUTTONUP:
            self.is_dragging_corner = False
            self.is_dragging_box = False
            self.zoom_center = (x, y)  # Used for Windows zooming in set_keys().

    def is_point_inside_box(self, point: tuple, box_points: np.ndarray) -> bool:
        """
        Check if a clicked point is inside a box using the ray casting algorithm.
        Args:
            point: The (x, y) cursor coordinate to check.
            box_points: The four corner points of a box.
        Returns:
            bool: True if the point is inside the box, False otherwise.
        """

        x, y = point

        # Need to be able to drag the drag corner of an active box without
        #  moving an underlying box. If there is overlap, then return False
        #  to prevent the inactive underlying box from being moved.
        # The drag corner area is a min_dim square around the bottom-right
        #  corner and should match that used in handle_mouse_events() for
        #  cv2.EVENT_LBUTTONDOWN.
        min_drag_size = self.min_dim // 2
        if self.active_box:
            corner_x, corner_y = self.active_box.points[2]
            if abs(corner_x - x) <= min_drag_size and abs(corner_y - y) <= min_drag_size:
                return False

        return cv2.pointPolygonTest(contour=box_points, pt=(x, y), measureDist=False) >= 0

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
            if len(_box.points) == 4 and self.is_point_inside_box(point,
                                                                  _box.points):
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
                    thickness=self.image_info['line thickness'] // 2,
                    color=self.cv_font_color['black'],
                    lineType=cv2.LINE_AA)

    def on_save(self):
        """
        Save to a text file boxes class index and point coordinates
        in YOLO OBB format. Also save the drawn image as a jpeg file.
        Called from the 'Save' button in YoloOBBControl.config_control_window()
        """
        with self.control_lock:
            # Each time 'n' is pressed, an empty points array is appended to
            #  the boxes list before any box is drawn. Multiple 'n' presses
            #  will append multiple 2-point 0,0 arrays to it. So, need to remove
            #  all non-4-point elements for an accurate count of drawn boxes.
            self.boxes = [_box for _box in self.boxes if len(_box.points) == 4]

            if not self.boxes:
                app.info_txt.set('No boxes to save.')
                # print('No boxes to save.')
                return

            # Create the results directory if it doesn't exist.
            Path('results').mkdir(parents=True, exist_ok=True)

            result_image = self.image_array.copy()
            img_name = self.image_info['short name']

            # Box.points are in absolute coordinates. So, need to convert pixels to
            #  normalized coordinates (0.0 to 1.0).
            # Write the data to a text file as yolo-obb labels and draw boxes on the image.
            with open(f"results/{img_name}.txt", "w") as file:
                for _box in self.boxes:
                    if len(_box.points) == 4:
                        obb_label = Utility.convert_to_obb_label_format(
                            box=_box, im_size=self.image_info['h&w'])
                        file.write(f'{obb_label}\n')

                        pts = np.array(_box.points, np.int32).reshape((-1, 1, 2))
                        cv2.polylines(result_image,[pts],
                                      isClosed=True,
                                      color=self.cv_font_color['orange'],
                                      thickness=self.image_info['line thickness'])
                        cv2.circle(result_image,
                                   center=_box.points[2],
                                   radius=self.get_circle_radius(),
                                   color=self.cv_font_color['orange'],
                                   thickness=-1)
                        self.put_text_class_index(result_image,
                                                  class_idx=str(_box.class_index),
                                                  box_points=_box.points)

            cv2.imwrite(f"results/{img_name}_result.jpg", result_image)

            app.info_txt.set(f'{len(self.boxes)} YOLO OBB labels, and the annotated image,\n'
                             ' were saved to the results folder.')
        # Need to provide a session record of save actions for the user.
        print(f'{len(self.boxes)} YOLO OBB labels, and their annotated image,'
              ' were saved to the results folder.')


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
        self.line_thickness_label = tk.Label()

        # Want the 'Get new image' button bg to match the image name label fg
        #  when focusOut. When focusIn, image name label fg uses a better contrast.
        #  The color values are X11 color names recognized by tkinter.
        self.color = {
            'img label': 'gold',
            'window': 'gray75',
            'dark': 'gray30',
            'save button': 'dodger blue',
            'line button': 'dark orange',
            'black': 'gray0',
        }

    def set_color_focusout(self):
        self.config(bg=self.color['dark'])
        self.entry_label.configure(bg=self.color['dark'])
        self.info_label.config(bg=self.color['dark'], fg=self.color['dark'],)
        self.img_name_label.config(bg=self.color['dark'], fg=self.color['img label'])
        self.line_thickness_label.config(bg=self.color['dark'], fg=self.color['line button'])

    def set_color_focusin(self):
        self.config(bg=self.color['window'],)
        self.entry_label.configure(bg=self.color['window'],)
        self.info_label.config(bg=self.color['window'], fg=self.color['black'],)
        self.img_name_label.config(bg=self.color['window'], fg=self.color['save button'])
        self.line_thickness_label.config(bg=self.color['window'], fg='black')

    def config_control_window(self):
        self.title('YOLO OBB Control')
        self.attributes("-topmost", True)
        self.geometry('390x255+100+400')  # width x height + x_offset + y_offset
        self.resizable(width=False, height=False)
        self.config(borderwidth=6, relief='groove')
        self.protocol('WM_DELETE_WINDOW', self.on_close)

        self.grid_columnconfigure(0,weight=1)
        self.grid_rowconfigure(list(range(7)), weight=1)  # Assuming 6 rows

        self.entry_label.config(text='Enter a YOLO class index number for the next box:')
        self.current_image_name.set(f'Current image: {Path(box_drawer.image_path).stem}')
        self.img_name_label.config(textvariable=self.current_image_name,)
        self.line_thickness_label.config(text='Line thickness',)

        # This text will be replaced with save metrics when the user saves.
        self.info_txt.set(f"A yolo labels file in the"
                          f" {box_drawer.labels_folder} folder\n"
                 'can draw those boxes on its corresponding image.')
        self.info_label.config(
            textvariable=self.info_txt,
            width=65,  # character width is a function of font size and platform
            font='TkTooltipFont',  # TkSmallCaptionFont
        )

        self.class_entry.config(width=3)
        self.class_entry.insert(index=tk.INSERT, string='0')  # Default value for class index.

        get_image_btn = tk.Button(master=self,
                                  text='Load new image file',
                                  command=box_drawer.open_image,
                                  background=self.color['img label'],
                                  )
        save_button = tk.Button(master=self,
                                text='Save to results',
                                command=box_drawer.on_save,
                                background=self.color['save button'],
                                )
        increase_thickness_btn = tk.Button(
            master=self,
            text='＋',  # Full-width plus sign, from https://coolsymbol.com/
            command=Utility.increase_line_thickness,
            background=self.color['line button'],
        )
        decrease_thickness_btn = tk.Button(
            master=self,
            text='－',  # Full-width minus sign, from https://coolsymbol.com/
            command=Utility.decrease_line_thickness,
            background=self.color['line button'],
        )

        self.bind('<Escape>', lambda _: self.on_close())
        self.bind("<FocusIn>", lambda _: self.set_color_focusin())
        self.bind("<FocusOut>", lambda _: self.set_color_focusout())
        self.class_entry.bind("<Return>", lambda _: self.set_class_index())
        self.class_entry.bind("<FocusOut>", lambda _: self.set_class_index())

        self.entry_label.grid(
            row=0, column=0,
            padx=10, pady=(5, 0), sticky=tk.EW)
        self.class_entry.grid(
            row=1, column=0,
            pady=5)
        self.img_name_label.grid(
            row=2, column=0,
            padx=10, pady=0, sticky=tk.EW)
        get_image_btn.grid(
            row=3, column=0,
            padx=10, pady=10,)
        self.info_label.grid(
            row=4, column=0,
            padx=10, pady=(0, 10), sticky=tk.EW)
        save_button.grid(
            row=5, column=0,
            padx=10, pady=5, sticky=tk.EW)
        increase_thickness_btn.grid(
            row=6, column=0,
            padx=(0, 90), pady=10, sticky=tk.E)  # to the right
        decrease_thickness_btn.grid(
            row=6, column=0,
            padx=(0, 45), pady=10, sticky=tk.E)  # less to the right
        self.line_thickness_label.grid(
            row=6, column= 0,
            sticky=tk.EW)  # centered

    def set_class_index(self):
        """
        Sets Box.class_index for the current box. This is the index of
        the class label for the box, as required for YOLO OBB label data.
        Called from class_entry.bind() methods.
        """

        # Force positive integer class index when loose focus, for when
        #  negative values and letters may be showing in the focused Entry field.
        try:
            class_index = abs(int(self.class_entry.get()))
            box_drawer.current_class_index = class_index
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

        with open(labels_path, 'r') as labels:
            labels_to_convert: list = labels.readlines()

        if not labels_to_convert:
            messagebox.showerror(
                title='Empty file',
                message=f'{Path(labels_path).name} is empty.',
                detail='Please select a valid YOLO labels file.'
            )
            return None

        for line, label in enumerate(labels_to_convert):
            for dat in label.split()[1:]:
                if not label.split()[0].isdigit():
                    messagebox.showerror(
                        title='Bad class index',
                        detail='The class index must be an integer.\n'
                               f'Please check line: {line +1}'
                    )
                    return None

        # Now check the first line for expected number of data elements.
        if not len(labels_to_convert[0].split()) in (5, 9):
            messagebox.showerror(
                title='Bad input format',
                detail=f'{Path(labels_path).name} cannot be used.\n'
                       'Expect either 5 or 9 data elements per line.\n'
                       'Only label data; no comments, no headers.\n'
                       'Object detection labels from yolo modeling:'
                       '  class_index center-x center-y w h\n'
                       'Labels from yolo-obb modeling:'
                       '  class_index x1 y1 x2 y2 x3 y3 x4 y4\n'
            )
            return None

        # Return the labels and the image h,w for conversion by view_labels().
        return labels_to_convert

    def on_close(self):
        """
        Allows clean closing or Tkinter and cv2 windows. Exit by using
        main window 'X' button or Esc key press.
        Called from YoloInfoWindow __init__() window protocol and button.
        """

        # Close in the reverse of the order they were opened.
        box_drawer.stop_loop.set()
        cv_thread.join()
        self.destroy()
        print('User quit the program.')
        sys.exit(0)


class Utility:
    """
    A class of static methods to provide utility functions.
    Methods include:
    view_labels()
    convert_from_obb_label_format()
    convert_to_obb_label_format()
    increase_line_thickness()
    decrease_line_thickness()
    show_help()
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
                detail=f'{len(box_drawer.boxes)} OBB boxes were created\n'
                       ' from the YOLO labels file.'
            )

    @staticmethod
    def convert_from_obb_label_format(pts: list,
                                      im_size: tuple) -> tuple:
        """
        Convert YOLO oriented bounding box (OBB)O labels to a format
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
        points = np.array(box.points)

        # Calculate center in pixels
        center_px = points.mean(axis=0)

        # Calculate primary axis vector (width direction)
        vec_w = points[1] - points[0]
        width_px = np.linalg.norm(vec_w)

        # Calculate perpendicular vector (height direction)
        vec_h = points[3] - points[0]
        height_px = np.linalg.norm(vec_h)

        # Calculate rotation angle (critical for reconstruction)
        angle_rad = np.arctan2(vec_w[1], vec_w[0])

        # Convert to normalized coordinates
        center_x_norm = center_px[0] / img_w
        center_y_norm = center_px[1] / img_h
        w_norm = width_px / img_w
        h_norm = height_px / img_h

        # Corners relative to center before rotation
        half_w, half_h = w_norm / 2, h_norm / 2
        corners = np.array([
            [-half_w, -half_h],
            [ half_w, -half_h],
            [ half_w,  half_h],
            [-half_w,  half_h]
        ])

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
        Increase the line thickness of the drawn boxes.
        """
        box_drawer.image_info['line thickness'] += 1

    @staticmethod
    def decrease_line_thickness():
        """
        Decrease the line thickness of the drawn boxes.
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
            thickness=1 #self.image_info['line thickness'],
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
        help_text.pack(fill=tk.BOTH, side=tk.LEFT, expand=True)


if __name__ == "__main__":

    # Instantiate the drawing class with the default image.
    # Loading with a starting image is necessary for flow architecture.
    box_drawer = BoxDrawer(image_path='images/start_image.jpg')

    # Create the tk.Tk control window as the main thread; via inheritance.
    app = YoloOBBControl()
    app.config_control_window()

    # Run update_image_info() after app Tk window is initialized because
    #  it uses Tk winfo_screenwidth() and winfo_screenheight().
    box_drawer.update_image_info()

    # Run live updates of cv2 image with a threaded while loop.
    #  Requires cv2.WINDOW_GUI_NORMAL instead of cv2.WINDOW_NORMAL.
    cv_thread = threading.Thread(target=box_drawer.draw_box)
    cv_thread.start()

    print(f'{Path(sys.modules["__main__"].__file__).stem} now running...')
    app.mainloop()
