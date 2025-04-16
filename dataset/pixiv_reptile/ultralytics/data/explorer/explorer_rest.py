# Ultralytics YOLO 🚀, AGPL-3.0 license

from io import BytesIO
from pathlib import Path
from typing import Any, List, Tuple, Union

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image
from tqdm import tqdm

from ultralytics.data.augment import Format
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.models.yolo.model import YOLO
from ultralytics.utils import LOGGER, USER_CONFIG_DIR, IterableSimpleNamespace, checks

from .utils import get_sim_index_schema, get_table_schema, plot_query_result, prompt_sql_query, sanitize_batch


class ExplorerDataset(YOLODataset):




class Explorer:














        sim_table.add(_yield_sim_idx())
        self.sim_index = sim_table
        return sim_table.to_pandas()

    def plot_similarity_index(self, max_dist: float = 0.2, top_k: float = None, force: bool = False) -> Image:
        """
        Plot the similarity index of all the images in the table. Here, the index will contain the data points that are
        max_dist or closer to the image in the embedding space at a given index.

        Args:
            max_dist (float): maximum L2 distance between the embeddings to consider. Defaults to 0.2.
            top_k (float): Percentage of closest data points to consider when counting. Used to apply limit when
                running vector search. Defaults to 0.01.
            force (bool): Whether to overwrite the existing similarity index or not. Defaults to True.

        Returns:
            (PIL.Image): Image containing the plot.

        Example:
            ```python
            exp = Explorer()
            exp.create_embeddings_table()

            similarity_idx_plot = exp.plot_similarity_index()
            similarity_idx_plot.show() # view image preview
            similarity_idx_plot.save('path/to/save/similarity_index_plot.png') # save contents to file
            ```
        """
        sim_idx = self.similarity_index(max_dist=max_dist, top_k=top_k, force=force)
        sim_count = sim_idx["count"].tolist()
        sim_count = np.array(sim_count)

        indices = np.arange(len(sim_count))

        # Create the bar plot
        plt.bar(indices, sim_count)

        # Customize the plot (optional)
        plt.xlabel("data idx")
        plt.ylabel("Count")
        plt.title("Similarity Count")
        buffer = BytesIO()
        plt.savefig(buffer, format="png")
        buffer.seek(0)

        # Use Pillow to open the image from the buffer
        return Image.fromarray(np.array(Image.open(buffer)))

    def _check_imgs_or_idxs(
        self, img: Union[str, np.ndarray, List[str], List[np.ndarray], None], idx: Union[None, int, List[int]]
    ) -> List[np.ndarray]:
        """Determines whether to fetch images or indexes based on provided arguments and returns image paths."""
        if img is None and idx is None:
            raise ValueError("Either img or idx must be provided.")
        if img is not None and idx is not None:
            raise ValueError("Only one of img or idx must be provided.")
        if idx is not None:
            idx = idx if isinstance(idx, list) else [idx]
            img = self.table.to_lance().take(idx, columns=["im_file"]).to_pydict()["im_file"]

        return img if isinstance(img, list) else [img]

    def ask_ai(self, query):
        """
        Ask AI a question.

        Args:
            query (str): Question to ask.

        Returns:
            (pandas.DataFrame): A dataframe containing filtered results to the SQL query.

        Example:
            ```python
            exp = Explorer()
            exp.create_embeddings_table()
            answer = exp.ask_ai('Show images with 1 person and 2 dogs')
            ```
        """
        result = prompt_sql_query(query)
        try:
            return self.sql_query(result)
        except Exception as e:
            LOGGER.error("AI generated query is not valid. Please try again with a different prompt")
            LOGGER.error(e)
            return None

    def visualize(self, result):
        """
        Visualize the results of a query. TODO.

        Args:
            result (pyarrow.Table): Table containing the results of a query.
        """
        pass

    def generate_report(self, result):
        """
        Generate a report of the dataset.

        TODO
        """
        pass