import os
import numpy as np
import pandas as pd
import kagglehub
from tqdm import tqdm

import constants as cst
class BTCDataBuilder:
    def __init__(
        self, 
        data_dir: str,
        sampling_time: cst.SamplingTime,
        dates: list[str],
        split_rates: list[float],
        split_days: list[int]
    ): 
        self.n_lob_levels = cst.N_LOB_LEVELS
        self.data_dir = data_dir
        self.sampling_time = sampling_time
        self.dates = dates
        self.split_rates = split_rates
        self.split_days = split_days
        self.train_set = None
        self.val_set = None
        self.test_set = None
        
    def prepare_and_save(self):
        save_dir = "{}/{}/from_{}_to_{}".format(
            self.data_dir,
            cst.DatasetType.BTC_USDT_SPOT.value,
            self.dates[0],
            self.dates[-1]
        )
        os.makedirs(save_dir, exist_ok=True)
        
        if len(os.listdir(save_dir)) == 0:
            print("Downloading BTC - USDT SPOT orderbook data from Kaggle ...")
            path = kagglehub.dataset_download(
                "macqueen01/btcusdt-orderbook", force_download=False)
            
            files = os.listdir(path)
            
            for file_path, file_name in tqdm(
                [(os.path.join(path, file), file) for file in files], 
                desc="Processing BTC orderbook files",
                unit="files"
            ):
                dataframe = pd.read_csv(file_path)
                # Gather timestamp, sell_price_#, sell_size_#, buy_price_#, buy_size_# columns
                # Create sell-buy pairs for each level: sell_price_i, sell_size_i, buy_price_i, buy_size_i
                columns = ['timestamp']
                for i in range(self.n_lob_levels):
                    columns += [
                        f"sell_price_{i+1}", f"sell_size_{i+1}",
                        f"buy_price_{i+1}", f"buy_size_{i+1}"
                    ]
                dataframe = dataframe.loc[:, columns]
                
                save_path = os.path.join(save_dir, file_name)            
                if os.path.exists(save_path): continue
                
                dataframe.to_csv(save_path, index=False)
            
        self.num_trading_days = len(os.listdir(save_dir))
        self.dataframes = []
        self._prepare_dataframes(save_dir)
        
        print(self.train_labels_horizons.values.shape)
        print(self.dataframes[0].values.shape)
        
        self.train_set = np.concatenate([self.dataframes[0].values, self.train_labels_horizons.values], axis=-1)
        self.val_set = np.concatenate([self.dataframes[1].values, self.val_labels_horizons.values], axis=-1)
        self.test_set = np.concatenate([self.dataframes[2].values, self.test_labels_horizons.values], axis=-1)
        
        path_where_to_save = "{}/{}".format(
            self.data_dir,
            "tlob_BTC_USDT_SPOT",
        )
        self._save(path_where_to_save)   
        
    def _prepare_dataframes(self, dir: str):
        train_set = None
        val_set = None
        test_set = None
        [train_days, val_days, test_days] = self._split_days()
        
        for i, (file_path, file_name) in enumerate([(os.path.join(dir, file_name), file_name) for file_name in os.listdir(dir)]):
            if not os.path.isfile(file_path):
                raise ValueError(f"File {file_path} is not a file")
            
            df_ob = pd.read_csv(file_path)
            print(f"{file_name} has shape: ", df_ob.shape)
            df_ob = self._sample(df_ob)
            df_ob.drop(columns=['timestamp'])
            
            # Issue: current data is not consecutive, so somehow we need to seperate...
            if i < train_days:
                train_set = df_ob if train_set is None else pd.concat([train_set, df_ob])
            elif i < val_days:
                val_set = df_ob if val_set is None else pd.concat([val_set, df_ob])
            else:
                test_set = df_ob if test_set is None else pd.concat([test_set, df_ob])

        if train_set is None or val_set is None or test_set is None:
            raise ValueError("Train, val or test set is None")
        
        self.dataframes = [train_set, val_set, test_set]
        
        for i, horizon in enumerate(cst.HORIZONS):
            train_labels = self._label_data(
                self.dataframes[0], cst.LEN_SMOOTH, horizon)
            val_labels = self._label_data(
                self.dataframes[1], cst.LEN_SMOOTH, horizon)
            test_labels = self._label_data(
                self.dataframes[2], cst.LEN_SMOOTH, horizon)
            
            if i == 0:
                self.train_labels_horizons = pd.DataFrame(train_labels, columns=[f"label_h{horizon}"], dtype=np.float64)
                self.val_labels_horizons = pd.DataFrame(val_labels, columns=[f"label_h{horizon}"], dtype=np.float64)
                self.test_labels_horizons = pd.DataFrame(test_labels, columns=[f"label_h{horizon}"], dtype=np.float64)
            else:
                self.train_labels_horizons[f"label_h{horizon}"] = train_labels.astype(np.float64)
                self.val_labels_horizons[f"label_h{horizon}"] = val_labels.astype(np.float64)
                self.test_labels_horizons[f"label_h{horizon}"] = test_labels.astype(np.float64)
        
        self._normalize_data()
        
    def _label_data(self, input: pd.DataFrame, window_size: int, horizon: int) -> np.ndarray:
        sell_price_over_window = np.lib.stride_tricks.sliding_window_view(
            input['sell_price_1'], window_shape=window_size)
        buy_price_over_window = np.lib.stride_tricks.sliding_window_view(
            input['buy_price_1'], window_shape=window_size)
        avg_price = (buy_price_over_window + sell_price_over_window) / 2
        smoothen_avg_price = avg_price.mean(axis=-1)
        
        previous_price = smoothen_avg_price[:-horizon]
        future_price = smoothen_avg_price[horizon:]
        
        percentage_change = (future_price - previous_price) / previous_price
        alpha = np.abs(percentage_change).mean()
        
        print(f"alpha is set to average percentage change: {alpha}")
        
        # UP: 0
        # STATIONARY: 1
        # DOWN: 2
        labels = np.where(
            percentage_change < -alpha, 2, 
            np.where(percentage_change > alpha, 0, 1)
        )
        len_empty_timestamps = input.values.shape[0] - labels.shape[0]
        return np.concatenate([labels, np.full(shape=(len_empty_timestamps), fill_value=np.inf)])

    def _normalize_data(self):
        train_mean_price, train_std_price = self._get_stats_for_price(self.dataframes[0])
        train_mean_size, train_std_size = self._get_stats_for_size(self.dataframes[0])
        
        for i in range(len(self.dataframes)):
            self.dataframes[i] = self._normalize_price(self.dataframes[i], train_mean_price, train_std_price)
            self.dataframes[i] = self._normalize_size(self.dataframes[i], train_mean_size, train_std_size)
    
    def _get_stats_for_size(self, dataframe: pd.DataFrame) -> tuple[float, float]:
        mean_sell_size: float = dataframe.loc[:, [f"sell_size_{j+1}" for j in range(self.n_lob_levels)]].stack().mean()  # type: ignore
        mean_buy_size: float = dataframe.loc[:, [f"buy_size_{j+1}" for j in range(self.n_lob_levels)]].stack().mean()  # type: ignore
        mean_size = (mean_sell_size + mean_buy_size) / 2
        
        std_sell_size: float = dataframe.loc[:, [f"sell_size_{j+1}" for j in range(self.n_lob_levels)]].stack().std()  # type: ignore
        std_buy_size: float = dataframe.loc[:, [f"buy_size_{j+1}" for j in range(self.n_lob_levels)]].stack().std()  # type: ignore
        std_size = (std_sell_size + std_buy_size) / 2
        
        return mean_size, std_size

    def _get_stats_for_price(self, dataframe: pd.DataFrame) -> tuple[float, float]:
        mean_sell_price: float = dataframe.loc[:, [f"sell_price_{j+1}" for j in range(self.n_lob_levels)]].stack().mean()  # type: ignore
        mean_buy_price: float = dataframe.loc[:, [f"buy_price_{j+1}" for j in range(self.n_lob_levels)]].stack().mean()  # type: ignore
        mean_price = (mean_sell_price + mean_buy_price) / 2
        
        std_sell_price: float = dataframe.loc[:, [f"sell_price_{j+1}" for j in range(self.n_lob_levels)]].stack().std()  # type: ignore
        std_buy_price: float = dataframe.loc[:, [f"buy_price_{j+1}" for j in range(self.n_lob_levels)]].stack().std()  # type: ignore
        std_price = (std_sell_price + std_buy_price) / 2
        
        return mean_price, std_price
    
    def _normalize_price(self, dataframe: pd.DataFrame, mean_price: float, std_price: float) -> pd.DataFrame:
        dataframe.loc[:, [f"sell_price_{j+1}" for j in range(self.n_lob_levels)]] = (dataframe.loc[:, [f"sell_price_{j+1}" for j in range(self.n_lob_levels)]] - mean_price) / std_price
        dataframe.loc[:, [f"buy_price_{j+1}" for j in range(self.n_lob_levels)]] = (dataframe.loc[:, [f"buy_price_{j+1}" for j in range(self.n_lob_levels)]] - mean_price) / std_price
        return dataframe
    
    def _normalize_size(self, dataframe: pd.DataFrame, mean_size: float, std_size: float) -> pd.DataFrame:
        dataframe.loc[:, [f"sell_size_{j+1}" for j in range(self.n_lob_levels)]] = (dataframe.loc[:, [f"sell_size_{j+1}" for j in range(self.n_lob_levels)]] - mean_size) / std_size
        dataframe.loc[:, [f"buy_size_{j+1}" for j in range(self.n_lob_levels)]] = (dataframe.loc[:, [f"buy_size_{j+1}" for j in range(self.n_lob_levels)]] - mean_size) / std_size
        return dataframe
        
    def _split_days(self) -> list[int]:
        train = self.split_days[0]
        val = self.split_days[1] + train
        test = self.split_days[2] + val
        print(f"There are {train} days for training, {val - train} days for validation and {test - val} days for testing")
        return [train, val, test]
    
    def _save(self, path_to_save: str):
        if (self.train_set is None or self.val_set is None or self.test_set is None):
            raise ValueError("Train, val or test set is None")
        
        if not os.path.exists(path_to_save):
            os.makedirs(path_to_save, exist_ok=True)
            
        np.save(path_to_save + "/train.npy", self.train_set)
        np.save(path_to_save + "/val.npy", self.val_set)
        np.save(path_to_save + "/test.npy", self.test_set)
        
    def _sample(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        # Convert milliseconds since epoch to datetime
        dataframe['timestamp'] = pd.to_datetime(dataframe['timestamp'], unit='ms', errors='coerce')
        return dataframe \
            .set_index('timestamp') \
            .resample(self.sampling_time.value).first() \
            .ffill() \
            .reset_index()
            
            