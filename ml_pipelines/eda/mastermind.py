import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from dython.nominal import associations


class EDAMastermind:
    def __init__(self, df: pd.DataFrame, hue: str = None):
        self.df = df
        self.cat_cols = self.df.select_dtypes(include=["object"]).columns
        self.num_cols = self.df.select_dtypes(include=["number"]).columns
        self.hue = hue

    def plot_missing_values(
        self, threshold: int = 0, figsize: tuple[int, int] = (12, 6)
    ):
        missing_values = self.df.isnull().mean() * 100
        missing_values = missing_values[missing_values > threshold].sort_values(
            ascending=False
        )
        plt.figure(figsize=figsize)
        sns.barplot(
            x=missing_values.index, y=missing_values.values, hue=missing_values.index
        )
        plt.xticks(rotation=90)
        plt.xlabel("Features")
        plt.ylabel("Percentage of Missing Values")
        plt.title(f"Missing Values Distribution in DataFrame (threshold = {threshold})")
        plt.show()

    def plot_pairplot(
        self, height: float = 1.5, aspect: float = 1.5, diag_kind: str = "kde"
    ):
        sns.pairplot(
            data=self.df,
            hue=self.hue,
            height=height,
            diag_kind=diag_kind,
            aspect=aspect,
        )
        plt.show()

    def plot_heatmap(
        self,
        figsize: tuple[int, int] = (15, 15),
        annot: bool = True,
        fmt: str = ".1f",
        linewidths: float = 0.5,
    ):
        asso_df = associations(self.df, nominal_columns="all", plot=False)
        corr_matrix = asso_df["corr"]
        plt.figure(figsize=figsize)
        sns.heatmap(corr_matrix, annot=annot, fmt=fmt, linewidths=linewidths)
        plt.title("Correlation Matrix including Categorical Features")
        plt.show()

    def plot_countsplots(self, figsize: tuple[int, int] = (10, 5)):
        for col in self.cat_cols:
            hue = self.hue if self.hue else col
            plt.figure(figsize=figsize)
            sns.countplot(x=col, hue=hue, data=self.df)
            plt.title(f"Countplot Column {col}")
        plt.show()

    def plot_piecharts(self, figsize: tuple[int, int] = (7, 7)):
        for col in self.cat_cols:
            counts = self.df[col].value_counts()
            plt.figure(figsize=(figsize))
            plt.pie(counts.values, labels=counts.index, autopct="%1.1f%%")
            plt.title(f"PieChart Column {col}")
        plt.show()

    def plot_boxplots(self, figsize: tuple[int, int] = (10, 6)):
        for col in self.num_cols:
            hue = self.hue if self.hue else col
            plt.figure(figsize=figsize)
            sns.boxplot(x=col, data=self.df, hue=hue)
            plt.title(f"Boxplot Column {col}")
        plt.show()

    def plot_histograms(self, kde: bool = True, height: int = 5, aspect: int = 2):
        for col in self.num_cols:
            hue = self.hue if self.hue else col
            sns.displot(
                data=self.df, x=col, kde=kde, height=height, aspect=aspect, hue=hue
            )
            plt.title(f"Histogram Column {col}")
        plt.show()

    def plot_class_balance(self, figsize: tuple[int, int] = (10, 6)):
        plt.figure(figsize=figsize)
        sns.countplot(x=self.hue, hue=self.hue, data=self.df)
        plt.title(f"Class Balance in Column: {self.hue}", fontsize=16)
        plt.xlabel(self.hue, fontsize=14)
        plt.ylabel("Count", fontsize=14)
        plt.tight_layout()
        plt.show()
