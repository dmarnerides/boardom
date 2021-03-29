import time
import warnings
import matplotlib
import matplotlib.pyplot as plt
import boardom as bd

## https://stackoverflow.com/questions/22873410/how-do-i-fix-the-deprecation-warning-that-comes-with-pylab-pause
warnings.filterwarnings("ignore", ".*GUI is implemented.*")

## https://stackoverflow.com/questions/45729092/make-interactive-matplotlib-window-not-pop-to-front-on-each-update-windows-7
def mypause(interval):
    backend = matplotlib.rcParams['backend']
    if backend in matplotlib.rcsetup.interactive_bk:
        figManager = matplotlib._pylab_helpers.Gcf.get_active()
        if figManager is not None:
            canvas = figManager.canvas
            if canvas.figure.stale:
                canvas.draw()
            canvas.start_event_loop(interval)
            return

    # No on-screen figure is active, so sleep() is all we need.
    time.sleep(interval)


_pause_min = 1 / 60


def imshow(
    img,
    view=None,
    figure=None,
    pause=0,
    title=None,
    interactive=False,
    make_grid=False,
    *args,
    **kwargs,
):
    """Displays a Tensor or Array image to screen.

    Args:
        img (Tensor or Array): Image to display.
        view (str, optional): View of image. For more details see
            :func:`bd.change_view` (default 'torch').
        figure (int, optional): Use selected figure (default None).
        pause (float, optional): Number of seconds to pause execution for
            displaying when interactive is True. If a value less than 1/60 is
            given then it defaults to 1/60 (default 1/60).
        title (str, optional): Title for figure (default None).
        interactive (bool, optional): If the image will be updated; uses
            plt.ion() (default False).
        *args (optional): Extra arguments to be passed to plt.imshow().
        **kwargs (optional): Extra keyword arguments to be passed to plt.imshow().
    Example:
        >>> for video_1_frame, video_2_frame in two_videos_frames:
        >>>     bd.imshow(video_1_frame, view='cv', figure=1, interactive=True, title='Video 1')
        >>>     bd.imshow(video_2_frame, view='cv', figure=2, interactive=True, title='Video 2')

    """
    view = view or bd.default_view
    if figure is None:
        if imshow.my_figure is None:
            imshow.my_figure = plt.figure().number
        figure = imshow.my_figure
    else:
        figure = plt.figure(figure).number

    if title is not None:
        f = plt.gcf()
        f.canvas.set_window_title(title)

    if interactive:
        plt.ion()
        plt.clf()
        pause = max(_pause_min, pause)
    else:
        plt.ioff()

    if make_grid:
        img = bd.make_grid(img, view=view)
    img = bd.to_array(img)
    if img.ndim not in (2, 3):
        raise ValueError(
            f'Images must have two or three dimensions. Got shape: {img.shape}'
        )

    img = bd.change_view(img, view, 'plt')
    if img.shape[-1] not in (1, 3, 4):
        raise ValueError(
            'Invalid number of channels ({0}). '.format(img.shape[-1])
            + 'Perhaps you used the wrong view?'
        )
    img = img.squeeze()

    plt.gcf().gca().set_position([0.0, 0.0, 1.0, 1.0])
    plt.axis('off')
    plt.imshow(img, cmap='gray' if img.ndim == 2 else None, *args, **kwargs)

    if pause > 0:
        mypause(pause)

    if figure not in imshow.displayed or not interactive:
        imshow.displayed.add(figure)
        plt.show(block=not interactive)

    return figure


imshow.my_figure = None
imshow.displayed = set()
