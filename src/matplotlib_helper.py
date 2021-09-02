import array
import copy
from typing import Optional

import astropy.wcs as WCS
import matplotlib as mpl
import matplotlib.axes as axes
import matplotlib.pyplot as plt
import numpy as np
import pyregion
import scipy.odr as odr
import yaml
from astropy.io import fits
from matplotlib.artist import Artist
from matplotlib.legend_handler import HandlerTuple
from matplotlib.offsetbox import AnchoredText
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from scipy.stats import gaussian_kde
from scipy.stats.stats import pearsonr, spearmanr

import src.helper as helper
from src.PrecisionScalarFormatter import PrecisionScalarFormatter


def setup_matploblib(magnetic: bool):
    """Setup default plot layout

    :param magnetic: Switch whether the layout should be for magnetic maps.
    """
    # set some aesthetic plot parameters
    plt.style.use(["seaborn-deep", "./config/plots.mplstyle"])
    TINY_SIZE = 6
    SMALL_SIZE = 7
    MEDIUM_SIZE = 9
    BIGGER_SIZE = 10

    TEXT_WIDTH = 0.5

    width = TEXT_WIDTH * 6.202
    height = TEXT_WIDTH * 5.86

    plt.rc("font", size=MEDIUM_SIZE)  # controls default text sizes
    # plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    # plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    # plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    # plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    # plt.rc('legend', fontsize=TINY_SIZE)  # legend fontsize
    # plt.rc('legend', title_fontsize=MEDIUM_SIZE)
    # plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    plt.rcParams["image.cmap"] = "viridis" if not magnetic else "inferno"
    plt.rcParams["figure.figsize"] = [width, height]


def add_inline_title(title: str) -> Artist:
    """Add an inline title that imitates the look of a legend for overlays

    Args:
        title (str): title string (eg. the galaxy name)

    Returns:
        Artist: the mpl artist to add to the plot
    """
    anchored_text = AnchoredText(
        rf"\textbf{{ {title} }}",
        loc="upper right",
        prop=dict(size=mpl.rcParams["legend.title_fontsize"]),
    )
    anchored_text.patch.set_boxstyle("round", pad=0, rounding_size=0.2)
    anchored_text.patch.set_alpha(mpl.rcParams["legend.framealpha"])
    anchored_text.patch.set_edgecolor(mpl.rcParams["legend.edgecolor"])
    anchored_text.patch.set_facecolor(mpl.rcParams["legend.facecolor"])
    return anchored_text


def plot_magnetic(
    val: np.ndarray,
    wcs: WCS,
    output_path: str,
    region: pyregion.ShapeList,
    label: str = r"$B_{\mathrm{eq}}$ [\si{\micro G}]",
    vmin: float = 0,
    vmax: float = None,
    abs_val: bool = True,
    inline_title: str = None,
):
    """Plot and save a map of the magnetic field strength

    Args:
        val (np.ndarray): the magnetic field map (supplied in [G] or as relative and unitless)
        wcs (WCS): the wcs of the map
        output_path (str): where to save images,... (name without file extension)
        region (pyregion.ShapeList): region to plot into the image
        label (str, optional): label of the color bar. Defaults to r"{\mathrm{eq}}$ [\si{\micro G}]".
        vmin (float, optional): minimum value supplied to imshow. Defaults to 0.
        vmax (float, optional): maximum value supplied to imshow. Defaults to None.
        abs_val (bool, optional): switch between a relative magnetic field for eg. relative errors or absolute magnetic field strength. Defaults to True.
        inline_title (str, optional): inline title that will be shown inside the legend. Defaults to None.
    """
    ax = plt.axes(projection=wcs)

    ax.set_box_aspect(1)

    ax.add_artist(add_inline_title(inline_title))

    # Show magnetic field in µG
    im = ax.imshow(
        (val * 1e6) if abs_val else val,
        origin="lower",
        vmin=vmin,
        vmax=(vmax * 1e6) if abs_val and vmax else vmax,
    )

    ax.set_ylabel("Declination (J2000)")
    ax.set_xlabel("Right Ascension (J2000)")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05, axes_class=axes.Axes)

    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label(label)
    cbar.ax.tick_params(which="both", direction="out")
    cbar.minorticks_off()
    ax.grid(False)

    if region:
        patch_list = region.get_mpl_patches_texts()[0]
        for p in patch_list:
            ax.add_patch(p)

    # Write output files
    output_hdu = fits.PrimaryHDU(val)
    output_hdu.header.update(wcs.to_header())
    output_hdul = fits.HDUList(output_hdu)
    output_hdul.writeto(output_path + ".fits", overwrite=True)
    print("Out:", output_path + ".fits")

    # Extra room for title
    plt.savefig(output_path + ".png")
    print("Out:", output_path + ".png")
    plt.savefig(output_path + ".pdf")
    print("Out:", output_path + ".pdf")
    close_mpl_object()


def plot_magnetic_overlay(
    base: np.ndarray,
    overlay: np.ndarray,
    wcs: WCS,
    output_path: str,
    region: pyregion.ShapeList = None,
    overlay_label: str = r"$B_{\mathrm{eq}}$ [\si{\micro G}]",
    levels: Optional[np.ndarray] = None,
    inline_title: str = None,
):
    """Generate and save an overlay plot over an rgb base image (only the overlay will have a colorbar)

    Args:
        base (np.ndarray): base image
        overlay (np.ndarray): overlayed image (mostly the magnetic field strength)
        wcs (WCS): wcs to use for projection, should be the one of the base image
        output_path (str): where to save images,... (name without file extension)
        region (pyregion.ShapeList, optional): region to plot into the image. Defaults to None.
        overlay_label (str, optional): Label for the overlay colorbar. Defaults to r"{\mathrm{eq}}$ [\si{\micro G}]".
        levels (Optional[np.ndarray], optional): Levels for the overlayed image. Defaults to None.
        inline_title (str, optional): inline title that will be shown inside the legend. Defaults to None.
    """
    ax = plt.axes(projection=wcs)

    ax.set_box_aspect(1)

    extent = 0, overlay.shape[0], 0, overlay.shape[1]
    ax.imshow(base, origin="lower", cmap="gray", extent=extent)

    ax.add_artist(add_inline_title(inline_title))

    if region:
        patch_list = region.get_mpl_patches_texts()[0]
        for p in patch_list:
            ax.add_patch(p)

    cmap_overlay = list(plt.cm.inferno(np.linspace(0, 1, len(levels) + 4)))
    cmap_overlay = cmap_overlay[2:-2]

    cmap_overlay = mpl.colors.ListedColormap(cmap_overlay, "", len(cmap_overlay))
    norm_overlay = mpl.colors.BoundaryNorm(levels, cmap_overlay.N, extend="max")

    ax.contour(
        overlay,
        cmap=cmap_overlay,
        origin="lower",
        levels=levels,
        norm=norm_overlay,
        extend="both",
    )

    ax.set_ylabel("Declination (J2000)")
    ax.set_xlabel("Right Ascension (J2000)")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", "5%", pad=0.05, axes_class=axes.Axes)

    cbar = plt.colorbar(
        mpl.cm.ScalarMappable(norm=norm_overlay, cmap=cmap_overlay),
        cax=cax,
        orientation="vertical",
    )

    cbar.set_label(overlay_label)
    # Move the scientific notation string to the right side, otherwise it
    # overlaps with the second colorbar
    cbar.ax.yaxis.get_offset_text().set_position((4.8, 0))
    cbar.ax.tick_params(which="both", width=0, length=0)
    ax.tick_params(which="both", color="white")
    ax.grid(False)

    plt.savefig(output_path + ".png")
    print("Out:", output_path + ".png")
    plt.savefig(output_path + ".pdf")
    print("Out:", output_path + ".pdf")
    close_mpl_object()


def plot_overlay(
    base: np.ndarray,
    overlay: np.ndarray,
    base_label: str,
    wcs: WCS,
    output_path: str,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    region: pyregion.ShapeList = None,
    overlay_label: str = r"$B_{\mathrm{eq}}$ [\si{\micro G}]",
    levels: Optional[np.ndarray] = None,
    inline_title: str = None,
):
    """Generate and save an overlay plot over an rgb base image (only the overlay will have a colorbar)

    Args:
        base (np.ndarray): base image
        overlay (np.ndarray): overlayed image (mostly the magnetic field strength)
        base_label (str): label for the base colorbar
        wcs (WCS): wcs to use for projection, should be the one of the base image
        output_path (str): where to save images,... (name without file extension)
        vmin (Optional[float]): vmin for base image colors. Defaults to None.
        vmax (Optional[float]): vmax for base image colors. Defaults to None.
        region (pyregion.ShapeList, optional): region to plot into the image. Defaults to None.
        overlay_label (str, optional): Label for the overlay colorbar. Defaults to r"{\mathrm{eq}}$ [\si{\micro G}]".
        levels (Optional[np.ndarray], optional): Levels for the overlayed image. Defaults to None.
        inline_title (str, optional): inline title that will be shown inside the legend. Defaults to None.
    """
    ax = plt.axes(projection=wcs)

    ax.set_box_aspect(1)

    # Only positive values can be displayed
    base = abs(base)

    cmap_base = copy.copy(plt.cm.Greys)
    cmap_base.set_bad(color="white")
    # Use log scaling for the base image
    norm_base = mpl.colors.LogNorm(vmax=vmax, vmin=vmin, clip=False)

    im = ax.imshow(base, cmap=cmap_base, norm=norm_base, origin="lower")

    ax.add_artist(add_inline_title(inline_title))

    if region:
        patch_list = region.get_mpl_patches_texts()[0]
        for p in patch_list:
            ax.add_patch(p)

    cmap_overlay = list(plt.cm.inferno(np.linspace(0, 1, len(levels) + 4)))
    cmap_overlay = cmap_overlay[2:-2]

    cmap_overlay = mpl.colors.ListedColormap(cmap_overlay, "", len(cmap_overlay))
    norm_overlay = mpl.colors.BoundaryNorm(levels, cmap_overlay.N, extend="max")

    ax.contour(
        overlay,
        cmap=cmap_overlay,
        origin="lower",
        levels=levels,
        norm=norm_overlay,
        extend="both",
    )

    ax.set_ylabel("Declination (J2000)")
    ax.set_xlabel("Right Ascension (J2000)")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", "5%", pad=0.05, axes_class=axes.Axes)

    cbar = plt.colorbar(
        mpl.cm.ScalarMappable(norm=norm_overlay, cmap=cmap_overlay),
        cax=cax,
        orientation="vertical",
    )
    cbar.set_label(overlay_label)
    # Move the scientific notation string to the right side, otherwise it
    # overlaps with the second colorbar
    cbar.ax.yaxis.get_offset_text().set_position((4.8, 0))
    cbar.ax.tick_params(which="both", length=0, width=0)

    cax2 = divider.append_axes("top", size="5%", pad=0.05, axes_class=axes.Axes)

    # Logic to determine the vmin and vmax parameters
    if vmin is None:
        vmin = np.nanmin(base)

    if vmax is None:
        vmax = np.nanmax(base)

    # Try to get the precision that should be used for the ticks
    precision = int(np.ceil(-np.log10(vmin)))
    precision_s = precision - int(np.ceil(-np.log10(vmax)))
    if precision == 0:
        precision = 1
    ticks = np.around(
        np.logspace(np.log10(vmin), np.log10(vmax), 5, endpoint=True), precision
    )

    gbar = plt.colorbar(
        im,
        ticks=ticks,
        cax=cax2,
        orientation="horizontal",
        extend="both",
        format=PrecisionScalarFormatter(precision=precision_s),
    )

    gbar.ax.tick_params(which="both", direction="out")
    gbar.ax.minorticks_off()

    gbar.set_label(base_label)
    cax2.xaxis.set_ticks_position("top")
    cax2.xaxis.set_label_position("top")

    ax.grid(False)

    plt.savefig(output_path + ".png")
    print("Out:", output_path + ".png")
    plt.savefig(output_path + ".pdf")
    print("Out:", output_path + ".pdf")
    close_mpl_object()


def plot_pixel_power_law(
    x: np.ndarray,
    x_error: np.ndarray,
    y: np.ndarray,
    y_error: np.ndarray,
    z: np.ndarray,
    xlabel: str,
    output_path: str,
    region_mask: np.ndarray = None,
    ylabel: str = r"$B_{\mathrm{eq}}$ [\si{\micro G}]",
    p0: list = None,
    extra_line_params: array = None,
    fit_extra_line: bool = False,
    extra_line_label: str = "equipartition",
    x_value: str = "x",
    y_unit: str = r"\micro G",
    x_unit: str = "",
    cutoff: float = 0,
    density_map: bool = False,
    inline_title: str = None,
    spix_cutoff: bool = False,
    spix_cutoff_reverse: bool = False,
) -> None:
    """Generate and save a pixel plot (y against x) assuming a power law.
    This will attempt to make a ODR fit to a power law and plot the resulting model in the
    final images. The fit values will be saved to output_path.yml

    Args:
        x (np.ndarray): values for the x-axis usually SFR, HI,...
        x_error (np.ndarray): errors for the x-axis usually SFR, HI,...
        y (np.ndarray): values for the y-axis
        y_error (np.ndarray): errors for the y-axis
        z (np.ndarray): z-values for color coding (spectral index)
        xlabel (str): label for x-axis
        output_path (str): where to save images,... (name without file extension)
        region_mask (np.ndarray, optional): boolean mask to mask x,y,z. Defaults to None.
        ylabel (str, optional): label for y-axis. Defaults to r"{\mathrm{eq}}$ [\si{\micro G}]".
        p0 (list, optional): initial guesses for ODR fit. Defaults to None.
        extra_line_params (array, optional): show an extra line in plot with these params. Defaults to None.
        fit_extra_line (bool, optional): if true not only show a extra line but fit the extra line in the amplitude (extra_line_params provides the inital guess and the power law exponent). Defaults to False.
        extra_line_label (str, optional): label for the extra line inside the legend. Defaults to "equipartition".
        x_value (str, optional): Latex formatted sign for the x-axis (eg. \Sigma_{SFR}). Defaults to "x".
        y_unit (str, optional): Latex formatted unit for y-axis. Defaults to r"\micro G".
        x_unit (str, optional): Latex formatted unit for x-axis. Defaults to "".
        cutoff (float, optional): lower cutoff for ODR fit (mostly for energy density). Defaults to 0.
        density_map (bool, optional): should a density map be calculated and color coded in the plot. Defaults to False.
        inline_title (str, optional): inline title to show inside the legend. Defaults to None.
        spix_cutoff (bool, optional): use only the data with <-0.65 spectral index. Defaults to False.
        spix_cutoff_reverse (bool, optional): use only the data with >-0.65 spectral index. Defaults to False.
    """
    # Don't use NAN values for calculation, otherwise the linear regression is broken
    mask = (
        ~np.isnan(x)
        & ~np.isnan(y)
        & ~(x <= 0)
        & ~(y <= 0)
        & ~np.isnan(x_error)
        & ~np.isnan(y_error)
    )
    if region_mask is not None:
        print("Using a region mask")
        mask = mask & region_mask

    if spix_cutoff:
        mask = mask & (z < -0.65)

    if spix_cutoff_reverse:
        mask = mask & (z >= -0.65)

    if len(x[mask]) <= 2 or len(y[mask]) <= 2:
        print("Too few data point for fit...")
        return

    if not x[mask].any() or not y[mask].any():
        print("No values with specified map, returning...")
        return

    log_x = np.log10(x[mask])
    log_y = np.log10(y[mask])
    # log_x_error = 1/np.log(10) * x_error[mask]/x[mask]
    log_mask = ~np.isnan(log_x) & ~np.isnan(log_x)
    if cutoff > 0:
        log_mask = log_mask & (log_x >= np.log10(cutoff))

    ax = plt.axes()

    ax.set_box_aspect(1)

    cmap = None
    norm = None
    bounds = None
    if density_map:
        # Calculate the point density
        xy = np.vstack([log_x, log_y])
        z_l = gaussian_kde(xy)(xy)
        cmap = "jet"
    else:
        # Calculate the color coding and limits
        z_l = z[mask]

        bounds = []
        colors = []
        if (z_l < -0.85).any():
            bounds.append(np.floor(np.nanmin(z_l) * 10) / 10)
            colors.append("#48196B")
        if (z_l < -0.65).any():
            bounds.append(-0.85)
            colors.append("#2F698D")
        bounds.append(-0.65)
        bounds.append(np.ceil(np.nanmax(z_l) * 10) / 10)
        colors.append("#2AB07E")

        bounds = np.around(bounds, 2)

        if spix_cutoff:
            bounds = bounds[:-1]
            colors = colors[:-1]
        cmap = mpl.colors.ListedColormap(colors, "", len(colors))

        norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend="neither")

    marker_style = ["v", "D", "^"]
    sctr = []
    # If we have multple bounds, add the corresponding data seperatly with different markers
    if bounds is not None:
        for i in range(bounds.size - 1):
            sctr.append(
                ax.scatter(
                    x[mask & (z >= bounds[i]) & (z < bounds[i + 1])],
                    y[mask & (z >= bounds[i]) & (z < bounds[i + 1])],
                    c=colors[i],
                    marker=marker_style[i],
                )
            )
    else:
        sctr.append(
            ax.scatter(
                x[mask], y[mask], c=z_l, cmap=cmap, norm=norm, label="observed data",
            )
        )

    # Use loglog scale
    ax.set_yscale("log")
    ax.set_xscale("log")
    # Save xlim and ylim for later
    ylim = ax.get_ylim()
    xlim = ax.get_xlim()

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cax,
        orientation="vertical",
        spacing="proportional",
        format=FuncFormatter(lambda s, _: f"{s: .2f}"),
    )

    if density_map:
        cbar.set_label("point density")
        # point density is unitless
        cbar.set_ticks([])
        cbar.minorticks_off()
    else:
        cbar.set_label(r"radio spectral index")
        cbar.ax.tick_params(which="both", length=0, width=0)

    powerlaw_model = odr.Model(helper.powerlaw)
    data = odr.RealData(
        x[mask & (x > cutoff)],
        y[mask & (x > cutoff)],
        sy=y_error[mask & (x > cutoff)],
        sx=x_error[mask & (x > cutoff)],
    )
    lODR = odr.ODR(data, powerlaw_model, beta0=p0, maxit=10000)
    # Run full ODR and not leastsquares fit
    lODR.set_job(fit_type=0)
    out = lODR.run()
    out.pprint()

    # calculate chi2
    # residuals = (y[mask & (x > cutoff)] - helper.powerlaw(out.beta, x[mask & (x > cutoff)]))
    # chi_arr = residuals / y_error[mask & (x > cutoff)]
    # chi2_red = (chi_arr ** 2).sum() / (len(x[mask & (x > cutoff)]) - len(out.beta))

    #  use the residual variance from ODR as red chi2
    chi2_red = out.res_var

    # Calculate the scatter in [dex]
    log_residuals = np.log10(y[mask & (x > cutoff)]) - np.log10(
        helper.powerlaw(out.beta, x[mask & (x > cutoff)])
    )
    scatter = np.sqrt(1 / (log_residuals.size - 1) * np.sum(log_residuals ** 2))

    # use the "correct" sd_beta
    out.sd_beta = np.sqrt(np.diag(out.cov_beta))

    r_value = pearsonr(log_x[log_mask], log_y[log_mask])
    spearman_r = spearmanr(log_x[log_mask], log_y[log_mask])

    if out.info >= 5:
        print("The ODR fit DIDN't converge, stopping!")
        close_mpl_object()
        return

    print(f"amplitude: {out.beta[0]: .6e} +/- {out.sd_beta[0]: .6e}")
    print(f"index: {out.beta[1]: .6e} +/- {out.sd_beta[1]: .6e}")
    print(
        f"correlation: {out.cov_beta[0][1] / (out.cov_beta[0][0] * out.cov_beta[1][1]) ** (1 / 2):.6f}"
    )
    print(f"pearsonr: {r_value[0]:.6f}")
    print(f"spearmanr: {spearman_r[0]:.6f}")
    print(f"chisq: {chi2_red:.6f}")
    print(f"scatter: {scatter:.6f}")

    # fit_label: str = fr"fitted power law: $\SI{{{out.beta[0]: 4.2e}}}{{{y_unit}}} \cdot \left(\frac{{{x_value}}}{{{x_unit}}}\right)^{{ {out.beta[1]: 4.2f} }}$"
    fit_label: str = fr"fitted power law: $\propto \left(\frac{{{x_value}}}{{{x_unit}}}\right)^{{ {out.beta[1]: 4.2f} }}$"

    plot_x = np.logspace(
        np.log10(np.min(x[mask])) - 3, np.log10(np.max(x[mask])) + 3, num=100
    )

    chisq_static = None
    # Add the extra line and fit if wanted
    if extra_line_params:
        # params is the amplitude for this power law model
        params = extra_line_params[0]
        fixed_exp_func = lambda amp, x: amp * np.power(x, extra_line_params[1])
        ddof = len(x[mask & (x > cutoff)])
        if fit_extra_line:
            ddof = len(x[mask & (x > cutoff)]) - 1
            fixed_exp_model = odr.Model(fixed_exp_func)
            fixed_exp_data = odr.RealData(
                x[mask & (x > cutoff)],
                y[mask & (x > cutoff)],
                sy=y_error[mask & (x > cutoff)],
                sx=x_error[mask & (x > cutoff)],
            )

            fixed_exp_ODR = odr.ODR(
                fixed_exp_data,
                fixed_exp_model,
                beta0=[extra_line_params[0]],
                maxit=10000,
            )
            # Run full ODR and not leastsquares fit
            fixed_exp_ODR.set_job(fit_type=0)
            fixed_exp = fixed_exp_ODR.run()
            if out.info >= 5:
                print("The fixed ODR fit DIDN'T converge, stopping!")
                return
            params = fixed_exp.beta
            chisq_static = fixed_exp.res_var
        else:
            # calculate chi2
            residuals = y[mask & (x > cutoff)] - fixed_exp_func(
                params, x[mask & (x > cutoff)]
            )
            chi_arr = residuals / y_error[mask & (x > cutoff)]
            chisq_static = (chi_arr ** 2).sum() / ddof

        # Try to center the extra line in the equal aspect box, without cutting off data
        ylim_new = fixed_exp_func(params, xlim)

        if ylim_new[0] > ylim[0]:
            xlim = ((params ** (-1) * ylim[0]) ** (1 / extra_line_params[1]), xlim[1])
        else:
            ylim = (ylim_new[0], ylim[1])

        if ylim_new[1] < ylim[1]:
            xlim = (xlim[0], (params ** (-1) * ylim[1]) ** (1 / extra_line_params[1]))
        else:
            ylim = (ylim[0], ylim_new[1])

        # Plot equipartition style line
        plt_extra = ax.plot(
            plot_x,
            fixed_exp_func(params, plot_x),
            "--",
            c="red",
            label=extra_line_label,
        )

    # Indicate a cutoff if wanted
    if cutoff > 0 and np.nanmin(x[mask]) < cutoff:
        ax.vlines(
            x=cutoff,
            ymin=ylim[0],
            ymax=ylim[1],
            label="fit cutoff",
            color="darkorange",
            ls=":",
        )

    # Plot fit
    plt_fit = ax.plot(
        plot_x,
        helper.powerlaw([out.beta[0], out.beta[1]], plot_x),
        "-",
        c="black",
        label=fit_label,
    )

    # get the confidence intervall
    cbl, cbu = helper.confindence_band(
        plot_x, x[mask], y[mask], out.beta, out.cov_beta, conf=0.95
    )

    # uncertainty lines (95% confidence)
    ax.fill_between(
        plot_x, cbl, cbu, color="gray", edgecolor=None, alpha=0.5,
    )

    # add labels
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    # Manually construct the legend to add all used marker styles
    l_plots = [plt_fit[0]]
    if extra_line_params:
        l_plots.append(plt_extra[0])
    l_plots.append(tuple(sctr))

    l_labels = [fit_label]
    if extra_line_params:
        l_labels.append(extra_line_label)
    # label for the data
    l_labels.append("observed data")

    ax.legend(
        l_plots,
        l_labels,
        title=rf"\textbf{{ {inline_title} }}" if inline_title else None,
        handler_map={tuple: HandlerTuple(ndivide=None)},
        markerscale=3.5,
    )

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    result_dict = {
        "amplitude": {"value": float(out.beta[0]), "std": float(out.sd_beta[0])},
        "index": {"value": float(out.beta[1]), "std": float(out.sd_beta[1])},
        "correlation_fit": float(
            out.cov_beta[0][1] / (out.cov_beta[0][0] * out.cov_beta[1][1]) ** (1 / 2)
        ),
        "pearsonr": float(r_value[0]),
        "spearmanr": float(spearman_r[0]),
        "chisq": float(chi2_red),
        "chisq_static": float(chisq_static),
        "scatter": float(scatter),
        "x": {"value": float(np.mean(x[mask])), "std": float(np.std(x[mask]))},
        "y": {"value": float(np.mean(y[mask])), "std": float(np.std(y[mask]))},
    }
    # save fit results
    with open(output_path + ".yml", "w") as file:
        yaml.dump(result_dict, file)
        print("Out:", output_path + ".yml")

    plt.savefig(output_path + ".png")
    print("Out:", output_path + ".png")
    plt.savefig(output_path + ".pdf")
    print("Out:", output_path + ".pdf")
    close_mpl_object()

    # if not spix_cutoff and not spix_cutoff_reverse:
    #     plot_pixel_power_law(x=x, x_error=x_error, y=y, y_error=y_error, z=z, xlabel=xlabel,
    #                          output_path=output_path + "_cutoff", region_mask=region_mask, ylabel=ylabel, p0=p0,
    #                          extra_line_params=extra_line_params, fit_extra_line=fit_extra_line,
    #                          extra_line_label=extra_line_label, x_value=x_value, y_unit=y_unit,
    #                          x_unit=x_unit, cutoff=cutoff,
    #                          use_integrated_spix=use_integrated_spix, density_map=density_map,
    #                          inline_title=inline_title, spix_cutoff=True)

    # if spix_cutoff and not spix_cutoff_reverse:
    #     plot_pixel_power_law(x=x, x_error=x_error, y=y, y_error=y_error, z=z, xlabel=xlabel,
    #                          output_path=output_path + "_reverse", region_mask=region_mask, ylabel=ylabel, p0=p0,
    #                          extra_line_params=extra_line_params, fit_extra_line=fit_extra_line,
    #                          extra_line_label=extra_line_label, x_value=x_value, y_unit=y_unit,
    #                          x_unit=x_unit, cutoff=cutoff,
    #                          use_integrated_spix=use_integrated_spix, density_map=density_map,
    #                          inline_title=inline_title, spix_cutoff=False, spix_cutoff_reverse=True)


def plot_pixel_mean_power_law(
    x: np.ndarray,
    y: np.ndarray,
    x_std: np.ndarray,
    y_std: np.ndarray,
    xlabel: str,
    output_path: str,
    ylabel: str = r"$\langle B_{\mathrm{eq}} \rangle$ [\si{\micro G}]",
    p0: list = None,
    extra_line_params: array = None,
    fit_extra_line: bool = False,
    extra_line_label: str = "equipartition",
    x_value: str = "x",
    y_unit: str = r"\micro G",
    x_unit: str = "",
    center_fixed: bool = False,
    no_mean: bool = False,
):

    """Generate and save a pixel plot for the mean values of all galaxies (y against x) assuming a power law.
    This will attempt to make a ODR fit to a power law and plot the resulting model in the
    final images. The fit values will be saved to output_path.yml

    Args:
        x (np.ndarray): values for the x-axis usually SFR, HI,...
        x_std (np.ndarray): standard deviation for the x-axis usually SFR, HI,...
        y (np.ndarray): values for the y-axis
        y_std (np.ndarray): standard deviations for the values on the y-axis
        xlabel (str): label for x-axis
        output_path (str): where to save images,... (name without file extension)
        ylabel (str, optional): label for y-axis. Defaults to r"{\mathrm{eq}}$ [\si{\micro G}]".
        p0 (list, optional): initial guesses for ODR fit. Defaults to None.
        extra_line_params (array, optional): show an extra line in plot with these params. Defaults to None.
        fit_extra_line (bool, optional): if true not only show a extra line but fit the extra line in the amplitude (extra_line_params provides the inital guess and the power law exponent). Defaults to False.
        extra_line_label (str, optional): label for the extra line inside the legend. Defaults to "equipartition".
        x_value (str, optional): Latex formatted sign for the x-axis (eg. \Sigma_{SFR}). Defaults to "x".
        y_unit (str, optional): Latex formatted unit for y-axis. Defaults to r"\micro G".
        x_unit (str, optional): Latex formatted unit for x-axis. Defaults to "".
        center_fixed (bool, optional): should the box aspect be centered on the extra line. Defaults to False.
        no_mean (bool, optional): should the fit labels include mean signs. Defaults to False.
    """
    log_x = np.log10(x)
    log_y = np.log10(y)

    ax = plt.axes()

    ax.set_box_aspect(1)

    # use loglog scale
    ax.set_yscale("log")
    ax.set_xscale("log")

    # this looks weird but seems to be needed
    if center_fixed:
        ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())

    ax.errorbar(
        x,
        y,
        xerr=x_std,
        yerr=y_std,
        fmt="v",
        c="#2F698D",
        elinewidth=0.8,
        markersize=2,
        label="observed data",
    )

    ylim = ax.get_ylim()
    xlim = ax.get_xlim()

    powerlaw_model = odr.Model(helper.powerlaw)
    data = odr.RealData(x, y, sy=y_std, sx=x_std)

    lODR = odr.ODR(data, powerlaw_model, beta0=p0, delta0=x_std, maxit=10000)
    # Run full ODR and not leastsquares fit
    lODR.set_job(fit_type=0)
    out = lODR.run()

    # calculate chi2
    # residuals = (y - helper.powerlaw(out.beta, x))
    # chi_arr = residuals / y_std
    # chi2_red = (abs(chi_arr)).sum() / (len(x) - len(out.beta))

    # calculate scatter in [dex]
    log_residuals: np.ndarray = (np.log10(y) - np.log10(helper.powerlaw(out.beta, x)))
    scatter = np.sqrt(1 / (log_residuals.size - 1) * np.sum(log_residuals ** 2))

    out.sd_beta = np.sqrt(np.diag(out.cov_beta))

    r_value = pearsonr(log_x, log_y)
    spearman_r = spearmanr(log_x, log_y)

    if int(out.info) >= 5:
        print("The ODR fit DIDN'T converge, stopping!")
        close_mpl_object()
        return

    print(f"amplitude: {out.beta[0]: .6e} +/- {out.sd_beta[0]: .6e}")
    print(f"index: {out.beta[1]: .6e} +/- {out.sd_beta[1]: .6e}")
    print(
        f"correlation: {out.cov_beta[0][1] / (out.cov_beta[0][0] * out.cov_beta[1][1]) ** (1 / 2):.6f}"
    )
    print(f"pearsonr: {r_value[0]:.6f}")
    print(f"spearmanr: {spearman_r[0]:.6f}")
    print(f"chisq: {out.res_var:.6f}")
    print(f"scatter: {scatter:.6f}")

    if no_mean:
        fit_label: str = fr"fitted power law: $ \propto \left(\frac{{ {x_value}}}{{{x_unit}}}\right)^{{ {out.beta[1]: 4.2f} }}$"
    else:
        fit_label: str = fr"fitted power law: $ \propto \left(\frac{{\langle {x_value}\rangle}}{{{x_unit}}}\right)^{{ {out.beta[1]: 4.2f} }}$"

    plot_x = np.logspace(np.log10(np.min(x)) - 3, np.log10(np.max(x)) + 3, num=100)

    chisq_static = None
    if extra_line_params:
        params = extra_line_params[0]
        fixed_exp_func = lambda amp, x: amp * np.power(x, extra_line_params[1])
        ddof = len(x)
        if fit_extra_line:
            ddof = len(x) - 1
            fixed_exp_model = odr.Model(fixed_exp_func)
            fixed_exp_data = odr.RealData(x, y, sy=y_std, sx=x_std)

            fixed_exp_ODR = odr.ODR(
                fixed_exp_data,
                fixed_exp_model,
                beta0=[extra_line_params[0]],
                maxit=10000,
            )
            # Run full ODR and not leastsquares fit
            fixed_exp_ODR.set_job(fit_type=0)
            fixed_exp = fixed_exp_ODR.run()
            fixed_exp.pprint()
            params = fixed_exp.beta
            chisq_static = fixed_exp.res_var

        if center_fixed:
            ylim_new = fixed_exp_func(params, xlim)

            if ylim_new[0] > ylim[0]:
                xlim = (
                    (params ** (-1) * ylim[0]) ** (1 / extra_line_params[1]),
                    xlim[1],
                )
            else:
                ylim = (ylim_new[0], ylim[1])

            if ylim_new[1] < ylim[1]:
                xlim = (
                    xlim[0],
                    (params ** (-1) * ylim[1]) ** (1 / extra_line_params[1]),
                )
            else:
                ylim = (ylim[0], ylim_new[1])

        # Plot extraline with different style
        ax.plot(
            plot_x,
            fixed_exp_func(params, plot_x),
            "--",
            c="red",
            label=extra_line_label,
        )

    cbl, cbu = helper.confindence_band(plot_x, x, y, out.beta, out.cov_beta, conf=0.95)

    # uncertainty lines (95% confidence)
    ax.fill_between(
        plot_x, cbl, cbu, color="gray", edgecolor=None, alpha=0.5,
    )

    # Plot fit
    ax.plot(
        plot_x,
        helper.powerlaw([out.beta[0], out.beta[1]], plot_x),
        "-",
        c="black",
        label=fit_label,
    )

    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    if center_fixed:
        ax.legend()
    else:
        ax.legend(loc="upper left")

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    result_dict = {
        "amplitude": {"value": float(out.beta[0]), "std": float(out.sd_beta[0])},
        "index": {"value": float(out.beta[1]), "std": float(out.sd_beta[1])},
        "correlation_fit": float(
            out.cov_beta[0][1] / (out.cov_beta[0][0] * out.cov_beta[1][1]) ** (1 / 2)
        ),
        "pearsonr": float(r_value[0]),
        "spearmanr": float(spearman_r[0]),
        "chisq": float(out.res_var),
        "chisq_fixed": float(chisq_static) if chisq_static else 0,
        "scatter": float(scatter),
        "x": {"value": float(np.mean(x)), "std": float(np.std(x))},
        "y": {"value": float(np.mean(y)), "std": float(np.std(y))},
    }
    with open(output_path + ".yml", "w") as file:
        yaml.dump(result_dict, file)
        print("Out:", output_path + ".yml")

    plt.savefig(output_path + ".png")
    print("Out:", output_path + ".png")
    plt.savefig(output_path + ".pdf")
    print("Out:", output_path + ".pdf")
    close_mpl_object()


def close_mpl_object():
    plt.clf()
    plt.cla()
    plt.close()