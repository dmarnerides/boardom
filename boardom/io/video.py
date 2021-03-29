import functools
from time import time
from weakref import WeakValueDictionary
import os
import boardom as bd
import cv2

# OpenCV Video IO flags:
# https://docs.opencv.org/3.4/d4/d15/group__videoio__flags__base.html
# Useful video codec / containers info
# https://en.wikipedia.org/wiki/Comparison_of_video_container_formats


class _Context:
    def open(self):
        pass

    def close(self):
        pass

    def __enter__(self):
        self.open()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class _DirectoryVideoReader(_Context):
    def __init__(self, directory, extension, load_fn=None):
        if load_fn is None:
            load_fn = bd.imread
        self.load_fn = load_fn
        self.directory = directory
        self.basename = os.path.basename(self.directory)
        self.extension = extension
        if not os.path.exists(self.directory):
            raise RuntimeError(f'Could not find frame directory: {self.directory}')
        filelist = [x for x in os.listdir(self.directory) if x.endswith(extension)]
        if not filelist:
            raise RuntimeError(
                f'Could not find {extension} video frames in directory: {self.directory}'
            )
        filelist = [x for x in filelist if x.startswith(self.basename)]
        if not filelist:
            raise RuntimeError(
                f'Could not find frames for {self.basename} video in directory: {self.directory}'
                f'\nNote: frames must match the pattern <dirname>/<dirname>_<frame>.<extension>'
            )
        filelist.sort(key=lambda x: int(x.split('.')[0].split('_')[-1]))
        self.filelist = [os.path.join(self.directory, x) for x in filelist]
        self.nframes = len(self.filelist)
        self.current_frame = 1

    def __iter__(self):
        # Refresh counter
        self.current_frame = 1
        return self

    def __next__(self):
        if self.current_frame < self.nframes:
            ret = self.filelist[self.current_frame]
            if self.load_fn is not None:
                ret = self.load_fn(ret)
            self.current_frame += 1
            return ret
        else:
            raise StopIteration

    def __len__(self):
        return self.nframes

    @bd.once_property
    def resolution(self):
        return None

    @bd.once_property
    def fps(self):
        return None


class _LDRVideoReader(_Context):
    def __init__(self, file):
        self.file = file
        self.capture = None
        self.context_count = 0
        if not os.path.exists(self.file):
            raise RuntimeError(f'Could not find video file: {self.file}')

    def _reset_capture(self):
        if self.capture is not None:
            self.capture.release()
        self.capture = cv2.VideoCapture(self.file)

    def open(self):
        self.context_count += 1
        if self.context_count == 1:
            self._reset_capture()

    def close(self):
        if (self.context_count == 1) and (self.capture is not None):
            self.capture.release()
        self.context_count -= 1
        self.capture = None

    def __del__(self):
        if self.capture is not None:
            self.capture.release()

    def __iter__(self):
        if self.context_count <= 0:
            raise RuntimeError(
                'Can only iterate Video inside a managed context: E.g. use like:'
                '\nwith reader:\n\tfor frame, i_frame in reader:\n\t\tpass\n'
            )
        # Reset the capture to start from the beginning
        self._reset_capture()
        return self

    def __next__(self):
        if (self.capture is not None) and self.capture.isOpened():
            ret, frame = self.capture.read()
            if not ret:
                raise StopIteration
            else:
                return frame
        else:
            bd.warn(
                'Something went wrong with video iteration. Perhaps context manager is inactive.'
            )
            StopIteration

    def __len__(self):
        if self.capture is not None:
            return int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        else:
            return 0

    @bd.once_property
    def resolution(self):
        with self:
            width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (width, height)

    @bd.once_property
    def fps(self):
        with self:
            fps = self.capture.get(cv2.CAP_PROP_FPS)
        return fps


class _LDRVideoWriter(_Context):
    # Resolution is (width, height)
    def __init__(self, file, fps, resolution):
        self.file = bd.process_path(file)
        self.extension = f'.{file.split(".")[-1]}'
        self.writer = None
        self.context_count = 0
        self.fourcc = Video.LDR_VID_EXT[self.extension]['fourcc']
        self.fps = float(fps)
        self.resolution = resolution
        directory = os.path.dirname(file)
        bd.process_path(directory, create=True)

    def open(self):
        self.context_count += 1
        if self.writer is None:
            self.writer = cv2.VideoWriter(
                self.file, self.fourcc, self.fps, self.resolution
            )

    def close(self):
        if (self.context_count == 1) and (self.writer is not None):
            self.writer.release()
        self.context_count -= 1
        self.writer = None

    def __del__(self):
        if self.writer is not None:
            self.writer.release()

    def write(self, frame):
        if (self.writer is None) or (self.context_count <= 0):
            raise RuntimeError(
                'Can only write Video inside a managed context: E.g. use like:'
                '\nwith writer:\n\twriter.write(frame)\n'
            )
        # Reset the writer to start from the beginning
        if self.writer.isOpened():
            self.writer.write(frame)
        else:
            bd.warn('Writer not opened.')

    __call__ = write


class _DirectoryVideoWriter(_Context):
    # Resolution is (width, height)
    def __init__(self, filename, extension):
        self.directory = bd.process_path(filename, create=True)
        self.fname = os.path.basename(filename)[: -len(extension)]
        self.extension = extension
        self.current_frame = 1

    def write(self, frame):
        fname = f'{self.fname}_{self.current_frame:09d}{self.extension}'
        bd.imwrite(frame, os.path.join(self.directory, fname))
        self.current_frame += 1

    __call__ = write


class ClassOnlyDescriptor:
    def __init__(self, name, value):
        self.name = name
        self.value = value

    def __get__(self, obj, cls):
        if obj is not None:
            raise AttributeError(f'{self.name} is not accessible from instances.')
        return self.value


# Good info on these here:
# https://en.wikipedia.org/wiki/Comparison_of_video_container_formats
class Video:
    _VIDEOS = ClassOnlyDescriptor('_VIDEOS', WeakValueDictionary())
    HDR_VID_EXT = {}
    HDR_IMG_EXT = {'.hdr': {}, '.pfm': {}, '.exr': {}}
    LDR_IMG_EXT = {'.jpg': {}, '.png': {}}
    LDR_VID_EXT = {
        #  '.avi': {'fourcc': cv2.VideoWriter_fourcc(*'H264')},
        '.avi': {'fourcc': cv2.VideoWriter_fourcc(*'DIVX')},
        '.mkv': {'fourcc': cv2.VideoWriter_fourcc(*'H264')},
        '.mp4': {'fourcc': cv2.VideoWriter_fourcc(*'mp4v')},
    }

    # Filename must be a filename with an extension
    # If the extension is not for ldr video then it's assumed that it's a directory
    # (the name without the extension)
    def __new__(cls, filename, fps=None, resolution=None):
        if not isinstance(filename, str):
            raise RuntimeError(
                f'Invalid filename, expected str type, got: {type(filename)}'
            )
        filename = bd.process_path(filename)
        all_extensions = (
            list(Video.HDR_VID_EXT.keys())
            + list(Video.HDR_IMG_EXT.keys())
            + list(Video.LDR_IMG_EXT.keys())
            + list(Video.LDR_VID_EXT.keys())
        )
        extension = '.' + filename.split(".")[-1]
        if extension not in all_extensions:
            raise RuntimeError(f'Unsupported extension: {extension}')
        is_ldr_video_file = extension in Video.LDR_VID_EXT

        _id = (filename, extension)
        if _id in Video._VIDEOS:
            return Video._VIDEOS[_id]
        self = super().__new__(cls)
        Video._VIDEOS[_id] = self
        self.is_ldr_video_file = is_ldr_video_file
        self.extension = extension
        self.filename = filename
        self.dirname = os.path.dirname(filename)
        self.fps = fps
        self.resolution = resolution
        return self

    @property
    def _id(self):
        return

    # load_fn applicable for directory videos
    @functools.lru_cache()
    def reader(self, load_fn=None):
        #  self._check_video_exists()
        if self.is_ldr_video_file:
            return _LDRVideoReader(self.filename)
        else:
            return _DirectoryVideoReader(self.dirname, self.extension, load_fn=load_fn)

    @functools.lru_cache()
    def writer(self, fps=None, resolution=None, extension=None):
        # fps, resolution needed for ldr video files
        fps = fps or self.fps
        resolution = resolution or self.resolution

        # extension is needed for directory files
        extension = extension or self.extension

        if self.is_ldr_video_file:
            for x, n in [(fps, 'fps'), (resolution, 'resolution')]:
                if x is None:
                    raise RuntimeError(f'{n} is required when writing LDR video')
            return _LDRVideoWriter(self.filename, fps, resolution)
        else:
            if extension is None:
                raise RuntimeError(
                    'extension is required when writing video frames in directory'
                )
            return _DirectoryVideoWriter(self.filename, extension)

    def show(self, fps=24, title=None):
        display = VideoDisplay(fps=fps, title=title)
        for frame in self.reader():
            display(frame)


class VideoDisplay:
    _VIDDISPLAYS = ClassOnlyDescriptor('_VIDDISPLAYS', WeakValueDictionary())

    def __new__(cls, fps=24, title=None):
        if not isinstance(fps, (float, int)):
            raise RuntimeError('fps must be float or int')
        fps = max(min(fps, 120), 0)
        if title is None:
            vid_id = len(VideoDisplay._VIDDISPLAYS) + 1
            title = f'Video {vid_id}'
        if title in VideoDisplay._VIDDISPLAYS:
            self = VideoDisplay._VIDDISPLAYS[title]
        else:
            self = super().__new__(cls)
            VideoDisplay._VIDDISPLAYS[title] = self
        self._title = title
        self._previous_frame_time = 0
        self._pause = 1 / fps
        return self

    def display(self, frame):
        sleep_secs = self._pause - time() + self._previous_frame_time
        cv2.waitKey(max(int(1000 * sleep_secs), 1))
        cv2.imshow(self._title, frame)
        self._previous_frame_time = time()

    __call__ = display


if __name__ == '__main__':
    # API example:

    reader = Video('fname.mp4').reader()

    out_resolution = (256, 256)
    writer = Video('out.mp4').writer(reader.fps, out_resolution)

    display = VideoDisplay(fps=reader.fps)

    # Context manager manages opening and closing of files
    with reader, writer:
        # Can iterate reader in a for loop and gives frames
        for frame in reader:
            # Example processecing of frame
            frame = cv2.resize(frame, out_resolution, interpolation=cv2.INTER_AREA)
            # Display frame (trying to keep the fps)
            display(frame)
            # Write to file
            writer(frame)
