import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output

class JupyterVisualizer:

    @classmethod
    def interactive_slice_viewer(self, data, axis=2, cmap='gray', title_prefix=""):
        """
        Create an interactive slice viewer with ipywidgets for Jupyter notebooks
        """
        vmin, vmax = np.percentile(data, [1, 99])

        # === Create slice slider controls ===
        max_slice = data.shape[axis] - 1
        slice_slider = widgets.IntSlider(
            value=max_slice // 2,
            min=0,
            max=max_slice,
            step=1,
            description=f'Slice:',
            continuous_update=False
        )

        window_min = widgets.FloatSlider(
            value=vmin,
            min=np.min(data),
            max=np.max(data),
            step=(np.max(data) - np.min(data)) / 100,
            description='Min:',
            continuous_update=False
        )

        window_max = widgets.FloatSlider(
            value=vmax,
            min=np.min(data),
            max=np.max(data),
            step=(np.max(data) - np.min(data)) / 100,
            description='Max:',
            continuous_update=False
        )

        out = widgets.Output()

        # === Update function ===
        def update(slice_idx, win_min, win_max):
            with out:
                clear_output(wait=True)

                if axis == 0:
                    slice_data = data[slice_idx, :, :]
                elif axis == 1:
                    slice_data = data[:, slice_idx, :]
                else:
                    slice_data = data[:, :, slice_idx]

                fig, ax = plt.subplots(figsize=(10, 8))
                im = ax.imshow(slice_data, cmap=cmap, vmin=win_min, vmax=win_max)
                plt.colorbar(im, ax=ax)
                ax.set_title(f'{title_prefix} Slice {slice_idx}/{max_slice} (axis {axis})')
                plt.tight_layout()
                plt.show()

        widgets.interactive_output(
            update,
            {'slice_idx': slice_slider, 'win_min': window_min, 'win_max': window_max}
        )
        controls = widgets.VBox([slice_slider, window_min, window_max])

        # display the widgets and output + init display
        display(widgets.HBox([controls, out]))
        update(slice_slider.value, window_min.value, window_max.value)


    @classmethod
    def interactive_ct_seg_viewer(self, ct_data, seg_data, axis=2, title_prefix=""):
        """
        Create an interactive viewer for CT with segmentation overlay
        """
        vmin, vmax = np.percentile(ct_data, [1, 99])

        # === Create slice slider controls ===
        max_slice = ct_data.shape[axis] - 1
        slice_slider = widgets.IntSlider(
            value=max_slice // 2,
            min=0,
            max=max_slice,
            step=1,
            description=f'Slice:',
            continuous_update=False
        )

        window_min = widgets.FloatSlider(
            value=vmin,
            min=np.min(ct_data),
            max=np.max(ct_data),
            step=(np.max(ct_data) - np.min(ct_data)) / 100,
            description='CT Min:',
            continuous_update=False
        )

        window_max = widgets.FloatSlider(
            value=vmax,
            min=np.min(ct_data),
            max=np.max(ct_data),
            step=(np.max(ct_data) - np.min(ct_data)) / 100,
            description='CT Max:',
            continuous_update=False
        )

        opacity_slider = widgets.FloatSlider(
            value=0.5,
            min=0,
            max=1.0,
            step=0.05,
            description='Seg Opacity:',
            continuous_update=False
        )

        out = widgets.Output()

        # === Update function ===
        def update(slice_idx, win_min, win_max, opacity):
            with out:
                clear_output(wait=True)

                if axis == 0:
                    ct_slice = ct_data[slice_idx, :, :]
                    seg_slice = seg_data[slice_idx, :, :]
                elif axis == 1:
                    ct_slice = ct_data[:, slice_idx, :]
                    seg_slice = seg_data[:, slice_idx, :]
                else:
                    ct_slice = ct_data[:, :, slice_idx]
                    seg_slice = seg_data[:, :, slice_idx]

                fig, ax = plt.subplots(figsize=(10, 8))

                # CT data
                ax.imshow(ct_slice, cmap='gray', vmin=win_min, vmax=win_max)

                # Overlay segmentation mask
                if np.any(seg_slice > 0):  # Only if there's actual segmentation
                    masked_seg = np.ma.masked_where(seg_slice == 0, seg_slice)
                    ax.imshow(masked_seg, cmap='hot', alpha=opacity)

                ax.set_title(f'{title_prefix} Slice {slice_idx}/{max_slice} (axis {axis})')
                plt.tight_layout()
                plt.show()

        widgets.interactive_output(
            update,
            {'slice_idx': slice_slider, 'win_min': window_min,
            'win_max': window_max, 'opacity': opacity_slider}
        )

        controls = widgets.VBox([slice_slider, window_min, window_max, opacity_slider])

        # display the widgets and output + init display
        display(widgets.HBox([controls, out]))
        update(slice_slider.value, window_min.value, window_max.value, opacity_slider.value)