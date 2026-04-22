from .ValidateReadVar import validate_read_variables, get_valid_variables
from .PlotHistogram import plot_stacked_hist, plot_histograms, histogram_2d
from .GetHistogram import get_histogram
from .PlotErrorBar import plot_errorbars
from .AnalysisParquet import analysis_parquet
from .DataSetsMagic import DIDS_DICT, VALID_SKIMS
from .GeneratedEvents import produced_event_count
from .ParquetDict import VALID_STR_CODE

__all__ = [
    "analysis_parquet",
    "analysis_uproot",
    "get_histogram",
    "get_valid_variables",
    "histogram_2d",
    "plot_errorbars",
    "plot_histograms",
    "plot_stacked_hist",
    "produced_event_count",
    "validate_read_variables",
    "DIDS_DICT",
    "VALID_SKIMS",
    "VALID_STR_CODE",
]


def __getattr__(name):
    if name == "analysis_uproot":
        from .AnalysisUproot import analysis_uproot

        return analysis_uproot
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
