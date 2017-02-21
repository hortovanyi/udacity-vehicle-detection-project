import typing
from sklearn.preprocessing import RobustScaler
from sklearn.svm import LinearSVC


class SearchParams(
    typing.NamedTuple(
        "GenSearchParams",
        [('color_space', str), ('orient', int), ('pix_per_cell', int),
         ('cell_per_block', int), ('hog_channel', int),
         ('spatial_size', tuple),
         ('hist_bins', int), ('hist_range', tuple),
         ('spatial_feat', bool),
         ('hist_feat', bool), ('hog_feat', bool),
         ('clf', LinearSVC), ('X_scaler', RobustScaler)
         ])):
    pass
