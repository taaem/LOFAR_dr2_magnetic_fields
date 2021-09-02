from matplotlib.ticker import ScalarFormatter


class PrecisionScalarFormatter(ScalarFormatter):
    def __init__(self, precision=3):
        super().__init__()
        self.precision = precision

    def _set_format(self):
        self.format = f"%1.{self.precision}f"
