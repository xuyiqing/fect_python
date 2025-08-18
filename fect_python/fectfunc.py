#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 25 22:11:43 2022

@author: liushijian
"""

import pandas as pd
import numpy as np
import rpy2
import rpy2.robjects as ro
from rpy2.robjects import Formula
import rpy2.rinterface as rinterface
from rpy2.robjects.packages import importr
import rpy2.robjects.conversion as cv

# for conversion
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.functions import SignatureTranslatedFunction

STM = SignatureTranslatedFunction
# for print and plot
grdevices = importr("grDevices")
import rpy2.ipython.html
from rpy2.ipython.ggplot import image_png

rprint = ro.globalenv.find("print")
rpy2.ipython.html.init_printing()

rgraphics = importr("graphics")


# %% import data and other fundamental function


def load_rdata(path: str, name=""):
    rsub_names = ro.r.load(
        path
    )  # now the datasets are in R env. This captures their names in rvector
    if name != "":
        with localconverter(ro.default_converter + pandas2ri.converter):
            return ro.conversion.rpy2py(ro.r[name])
    else:
        print("Please specify the dataset name: ", str(rsub_names))
        return


def _none2null(none_obj):  # converte python None into R NULL
    # return ro.r("NULL")
    return ro.NULL


def getCohort(data=None, D=None, index=None, varname=None, entry_time=None, **kwargs):
    rfect = importr("fect")
    _fec_getcohort = rfect._env[
        "get.cohort"
    ]  # find the get.cohort function in the fect env
    none_converter = cv.Converter("None converter")  # turn None into r_NULL
    none_converter.py2rpy.register(type(None), _none2null)
    if index != None:
        index = ro.StrVector(index)

    rlist = rinterface.baseenv[
        "list"
    ]  # to give the "list" input required for the function
    if entry_time != None:
        entry_list = []
        for sublist in entry_time:
            entry_list.append(ro.FloatVector(sublist))
        entry_time = rlist(*entry_list)

    with localconverter(
        ro.default_converter + pandas2ri.converter
    ):  # turn pandas df into rdf
        data = ro.conversion.py2rpy(data)

    with localconverter(ro.default_converter + none_converter):
        res = _fec_getcohort(
            data, D, index, varname, entry_time, **kwargs
        )  # a lot more to add here. plot.fect function

    with localconverter(ro.default_converter + pandas2ri.converter):
        res = ro.conversion.rpy2py(res)
    return res


# %% fect
# Or, we can define a fect Class. (not necessary because fect is just a dict)
def fect(
    formula: str,
    data=None,
    Y=None,
    D=None,
    X=None,
    group=None,
    na_rm=False,
    index=None,
    force="two-way",
    r=0,
    plambda=None,
    nlambda=10,
    CV=None,
    k=10,
    cv_prop=0.1,
    cv_treat=False,
    cv_nobs=3,
    cv_donut=0,
    criterion="mspe",
    binary=False,
    QR=False,
    method="fe",
    se=False,
    vartype="bootstrap",
    nboots=200,
    alpha=0.05,
    parallel=True,
    cores: int = None,
    tol=0.001,
    seed: int = None,
    min_T0: int = None,
    max_missing: int = None,
    proportion=0.3,
    pre_periods=None,
    f_threshold=0.5,
    tost_threshold=None,
    knots=None,
    degree=2,
    sfe=None,
    cfe=None,
    balance_period=None,
    fill_missing=False,
    placeboTest=False,
    placebo_period=None,
    carryoverTest=False,
    carryover_period=None,
    carryover_rm: int = None,
    loo=False,
    permute=False,
    m=2,
    normalize=False,
    **kwargs
):

    rfect = importr("fect")
    rfect.fect = STM(
        rfect.fect, init_prm_translate={"lambda": "plambda"}
    )  # call lambda as plambda
    formula = Formula(formula)  # get formula

    index = ro.StrVector(index)  # turn index into r_vector
    if knots != None:
        knots = ro.FloatVector(knots)
    if sfe != None:
        sfe = ro.StrVector(sfe)

    rlist = rinterface.baseenv[
        "list"
    ]  # to give the "list" input required for the function
    if cfe != None:
        cfe_list = []
        for sublist in cfe:
            cfe_list.append(ro.StrVector(sublist))
        cfe = rlist(*cfe_list)

    if balance_period != None:
        balance_period = ro.FloatVector(balance_period)
    if placebo_period != None:
        placebo_period = ro.FloatVector(placebo_period)
    if carryover_period != None:
        carryover_period = ro.FloatVector(carryover_period)
    if type(r) != int:
        r = ro.IntVector(r)

    with localconverter(
        ro.default_converter + pandas2ri.converter
    ):  # turn pandas df into rdf
        data = ro.conversion.py2rpy(data)

    none_converter = cv.Converter("None converter")  # turn None into r_NULL
    none_converter.py2rpy.register(type(None), _none2null)
    with localconverter(ro.default_converter + none_converter):

        res = rfect.fect(
            formula,
            data,
            Y,
            D,
            X,
            group,
            na_rm,
            index,
            force,
            r,
            plambda,
            nlambda,
            CV,
            k,
            cv_prop,
            cv_treat,
            cv_nobs,
            cv_donut,
            criterion,
            binary,
            QR,
            method,
            se,
            vartype,
            nboots,
            alpha,
            parallel,
            cores,
            tol,
            seed,
            min_T0,
            max_missing,
            proportion,
            pre_periods,
            f_threshold,
            tost_threshold,
            knots,
            degree,
            sfe,
            cfe,
            balance_period,
            fill_missing,
            placeboTest,
            placebo_period,
            carryoverTest,
            carryover_period,
            carryover_rm,
            loo,
            permute,
            m,
            normalize,
            **kwargs
        )
    return res
    # try form a dict for all para.


# %% PanelView


def panelview(
    formula: str = None,
    data=None,
    Y=None,
    D=None,
    X=None,
    index=None,
    ignore_treat=False,
    ptype="treat",
    outcome_type="continuous",
    treat_type=None,
    by_group=False,
    by_group_side=False,
    by_timing=False,
    theme_bw=True,
    xlim=None,
    ylim=None,
    xlab=None,
    ylab=None,
    gridOff=False,
    legendOff=False,
    legend_labs=None,
    main=None,
    pre_post=False,
    id=None,
    show_id=None,
    color=None,
    axis_adjust=False,
    axis_lab="both",
    axis_lab_gap=[0, 0],
    axis_lab_angle=None,
    shade_post=True,
    cex_main=15,
    cex_main_sub=12,
    cex_axis=8,
    cex_axis_x=None,
    cex_axis_y=None,
    cex_lab=12,
    cex_legend=12,
    background=None,
    style=None,
    by_unit=False,
    lwd=0.2,
    leave_gap=False,
    display_all=False,
    by_cohort=False,
    collapse_history=None,
    report_missing=False,
):
    print("Running panelview...")
    rpanelView = importr("panelView")
    rpanelView.panelview = STM(
        rpanelView.panelview, init_prm_translate={"type": "ptype"}
    )  # call type as ptype
    formula = Formula(formula)  # get formula

    index = ro.StrVector(index)  # turn index into r_vector
    if axis_lab_gap != None:
        axis_lab_gap = ro.FloatVector(axis_lab_gap)
    if xlim != None:
        xlim = ro.FloatVector(xlim)
    if ylim != None:
        ylim = ro.FloatVector(ylim)

    # if axis_adjust:
    #     axis_adjust = rinterface.BoolSexpVector((True,))  # turn into r vector
    # else:
    #     axis_adjust = rinterface.BoolSexpVector((False,))

    with localconverter(
        ro.default_converter + pandas2ri.converter
    ):  # turn pandas df into rdf
        data = ro.conversion.py2rpy(data)

    none_converter = cv.Converter("None converter")  # turn None into r_NULL
    none_converter.py2rpy.register(type(None), _none2null)
    with localconverter(ro.default_converter + none_converter):

        res = rpanelView.panelview(
            data,
            formula,
            Y,
            D,
            X,
            index,
            ignore_treat,
            ptype,
            outcome_type,
            treat_type,
            by_group,
            by_group_side,
            by_timing,
            theme_bw,
            xlim,
            ylim,
            xlab,
            ylab,
            gridOff,
            legendOff,
            legend_labs,
            main,
            pre_post,
            id,
            show_id,
            color,
            axis_adjust,
            axis_lab,
            axis_lab_gap,
            axis_lab_angle,
            shade_post,
            cex_main,
            cex_main_sub,
            cex_axis,
            cex_axis_x,
            cex_axis_y,
            cex_lab,
            cex_legend,
            background,
            style,
            by_unit,
            lwd,
            leave_gap,
            display_all,
            by_cohort,
            collapse_history,
            report_missing,
        )

    return res


# %% Plot Functions: The figure show altomatically in ipython, but need to plot again if want to show in py file.
def plt(
    fect_obj,
    ptype=None,
    loo="FALSE",
    highlight=None,
    plot_ci=None,
    show_points=None,
    show_group=None,
    bound=None,
    vis=None,
    count=True,
    proportion=0.3,
    pre_periods=None,
    f_threshold=None,
    tost_threshold=None,
    effect_bound_ratio=False,
    stats=None,
    stats_labs=None,
    main=None,
    xlim=None,
    ylim=None,
    xlab=None,
    ylab=None,
    gridOff=False,
    legendOff=False,
    legend_pos=None,
    legend_nrow=None,
    legend_labs=None,
    stats_pos=None,
    theme_bw=True,
    nfactors=None,
    include_FE=True,
    id=None,
    cex_main=None,
    cex_main_sub=None,
    cex_axis=None,
    cex_lab=None,
    cex_legend=None,
    cex_text=None,
    axis_adjust=False,
    axis_lab="both",
    axis_lab_gap=[0, 0],
    start0=False,
    return_test=False,
    balance=None,
    **kwargs
):

    if pre_periods != None:  # uniform type
        pre_periods = ro.FloatVector(pre_periods)
    if xlim != None:
        xlim = ro.FloatVector(xlim)
    if ylim != None:
        ylim = ro.FloatVector(ylim)
    if legend_labs != None:
        legend_labs = ro.StrVector(legend_labs)
    if stats_pos != None:
        stats_pos = ro.FloatVector(stats_pos)

    rfect = importr("fect")
    _fec_rprint = rfect._env["plot.fect"]  # find the fect.plot function in the fect env

    _fec_rprint = STM(
        _fec_rprint, init_prm_translate={"type": "ptype"}
    )  # call type as ptype

    none_converter = cv.Converter("None converter")  # turn None into r_NULL
    none_converter.py2rpy.register(type(None), _none2null)

    with localconverter(ro.default_converter + none_converter):
        res = _fec_rprint(
            fect_obj,
            ptype,
            loo,
            highlight,
            plot_ci,
            show_points,
            show_group,
            bound,
            vis,
            count,
            proportion,
            pre_periods,
            f_threshold,
            tost_threshold,
            effect_bound_ratio,
            stats,
            stats_labs,
            main,
            xlim,
            ylim,
            xlab,
            ylab,
            gridOff,
            legendOff,
            legend_pos,
            legend_nrow,
            legend_labs,
            stats_pos,
            theme_bw,
            nfactors,
            include_FE,
            id,
            cex_main,
            cex_main_sub,
            cex_axis,
            cex_lab,
            cex_legend,
            cex_text,
            axis_adjust,
            axis_lab,
            axis_lab_gap,
            start0,
            return_test,
            balance,
            **kwargs
        )  # a lot more to add here. plot.fect function
    return res


def png_show(fig):
    return image_png(fig)


class Fect:
    def __init__(self, fect_out=None, parameter=None, plot_para=None):
        self.fect_out = fect_out
        self.parameter = parameter
        self.plot_para = plot_para

    def getPara(self):
        return self.parameter

    def setPara(self, newpara):  # Do not recommand
        self.parameter = newpara

    def getFect(self):
        return self.fect_out

    # setFect will change every attribute of Fect
    def setFect(
        self,
        formula: str,
        data=None,
        Y=None,
        D=None,
        X=None,
        group=None,
        na_rm=False,
        index=None,
        force="two-way",
        r=0,
        plambda=None,
        nlambda=10,
        CV=None,
        k=10,
        cv_prop=0.1,
        cv_treat=False,
        cv_nobs=3,
        cv_donut=0,
        criterion="mspe",
        binary=False,
        QR=False,
        method="fe",
        se=False,
        vartype="bootstrap",
        nboots=200,
        alpha=0.05,
        parallel=True,
        cores: int = None,
        tol=0.001,
        seed: int = None,
        min_T0: int = None,
        max_missing: int = None,
        proportion=0.3,
        pre_periods=None,
        f_threshold=0.5,
        tost_threshold=None,
        knots=None,
        degree=2,
        sfe=None,
        cfe=None,
        balance_period=None,
        fill_missing=False,
        placeboTest=False,
        placebo_period=None,
        carryoverTest=False,
        carryover_period=None,
        carryover_rm: int = None,
        loo=False,
        permute=False,
        m=2,
        normalize=False,
        **kwargs
    ):
        self.fect_out = fect(
            formula,
            data,
            Y,
            D,
            X,
            group,
            na_rm,
            index,
            force,
            r,
            plambda,
            nlambda,
            CV,
            k,
            cv_prop,
            cv_treat,
            cv_nobs,
            cv_donut,
            criterion,
            binary,
            QR,
            method,
            se,
            vartype,
            nboots,
            alpha,
            parallel,
            cores,
            tol,
            seed,
            min_T0,
            max_missing,
            proportion,
            pre_periods,
            f_threshold,
            tost_threshold,
            knots,
            degree,
            sfe,
            cfe,
            balance_period,
            fill_missing,
            placeboTest,
            placebo_period,
            carryoverTest,
            carryover_period,
            carryover_rm,
            loo,
            permute,
            m,
            normalize,
            **kwargs
        )
        res = dict(self.fect_out.items())
        res = {k.replace(".", "_"): v for k, v in res.items()}
        del res["call"]  # this term is not needed but causes a lot of trouble.
        self.parameter = res
        self.plot_para = (
            None  # when new fect_out is assigned, plot_para needs to be regenerated.
        )

    def getPlot(self):
        return self.plot_para

    def setPlot(
        self,
        ptype=None,
        loo="FALSE",
        highlight=None,
        plot_ci=None,
        show_points=None,
        show_group=None,
        bound=None,
        vis=None,
        count=True,
        proportion=0.3,
        pre_periods=None,
        f_threshold=None,
        tost_threshold=None,
        effect_bound_ratio=False,
        stats=None,
        stats_labs=None,
        main=None,
        xlim=None,
        ylim=None,
        xlab=None,
        ylab=None,
        gridOff=False,
        legendOff=False,
        legend_pos=None,
        legend_nrow=None,
        legend_labs=None,
        stats_pos=None,
        theme_bw=True,
        nfactors=None,
        include_FE=True,
        id=None,
        cex_main=None,
        cex_main_sub=None,
        cex_axis=None,
        cex_lab=None,
        cex_legend=None,
        cex_text=None,
        axis_adjust=False,
        axis_lab="both",
        axis_lab_gap=[0, 0],
        start0=False,
        return_test=False,
        balance=None,
        **kwargs
    ):
        self.plot_para = plt(
            self.fect_out,
            ptype,
            loo,
            highlight,
            plot_ci,
            show_points,
            show_group,
            bound,
            vis,
            count,
            proportion,
            pre_periods,
            f_threshold,
            tost_threshold,
            effect_bound_ratio,
            stats,
            stats_labs,
            main,
            xlim,
            ylim,
            xlab,
            ylab,
            gridOff,
            legendOff,
            legend_pos,
            legend_nrow,
            legend_labs,
            stats_pos,
            theme_bw,
            nfactors,
            include_FE,
            id,
            cex_main,
            cex_main_sub,
            cex_axis,
            cex_lab,
            cex_legend,
            cex_text,
            axis_adjust,
            axis_lab,
            axis_lab_gap,
            start0,
            return_test,
            balance,
            **kwargs
        )


# %%  a new way to save. have't tried yet. worked well
def savefig(filepath: str = "fect_fig", fig_type: str = "png", width=512, height=512):
    if filepath.split(".")[-1] == fig_type:
        path = filepath
    else:
        path = filepath + "." + fig_type
    grdevices.dev_copy(ro.r[fig_type], file=path, width=width, height=height)
    grdevices.dev_off()
    return
