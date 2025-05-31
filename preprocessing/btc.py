from genericpath import isfile
import os
import random
import numpy as np
import pandas as pd
import kagglehub

import constants as cst

class BTCDataBuilder:
    def __init__(
        self, 
        data_dir: str,
        sampling_time: cst.SamplingTime,
        dates: list[str],
        split_rates: list[float]
    ): 
        self.n_lob_levels = cst.N_LOB_LEVELS
        self.data_dir = data_dir
        self.sampling_time = sampling_time
        self.dates = dates
        self.split_rates = split_rates
        self.train_set = None
        self.val_set = None
        self.test_set = None
        
    def prepare_and_save(self):
        # for date in self.dates:
        #     dataframe = pd.read_csv(self.data_dir + f"/{date}.csv")
        #     dataframe = self._sample(dataframe)
        
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
                "macqueen01/btcusdt-orderbook", cst.RAW_DATA_DIR, force_download=True)
            
            files = os.listdir(path)
            self.num_trading_days = len(files)
            
            for file_path in [os.path.join(path, file) for file in files]:
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
                dataframe.to_csv(os.path.join(save_dir, file_path), index=False)
            
        self.dataframes = []
        self._prepare_dataframes(save_dir)
        
    def _prepare_dataframes(self, dir: str):
        train_set = None
        val_set = None
        test_set = None
        [train_days, val_days, test_days] = self._split_days()
        
        files = [os.path.join(dir, file) for file in os.listdir(dir)]
        random.shuffle(files)
        
        for i, file in enumerate(files):
            if not os.path.isfile(file):
                raise ValueError(f"File {file} is not a file")
            
            df_ob = pd.read_csv(file)
            df_ob = self._sample(df_ob)
            df_ob.drop(columns=['timestamp'])
            
            if i < train_days:
                train_set = df_ob if train_set is None else pd.concat([train_set, df_ob])
            elif i < val_days:
                val_set = df_ob if val_set is None else pd.concat([val_set, df_ob])
            else:
                test_set = df_ob if test_set is None else pd.concat([test_set, df_ob])

        if train_set is None or val_set is None or test_set is None:
            raise ValueError("Train, val or test set is None")
        
        self.dataframes = [train_set, val_set, test_set]
        
        train_input = self.dataframes[0].values
        val_input = self.dataframes[1].values
        test_input = self.dataframes[2].values
        
        for i, horizon in enumerate(cst.HORIZONS):
            train_labels = self._label_data(
                train_input, cst.LEN_SMOOTH, horizon)
            val_labels = self._label_data(
                val_input, cst.LEN_SMOOTH, horizon)
            test_labels = self._label_data(
                test_input, cst.LEN_SMOOTH, horizon)
            
            if i == 0:
                self.train_labels_horizons = pd.DataFrame(train_labels, columns=[f"label_h{horizon}"])
                self.val_labels_horizons = pd.DataFrame(val_labels, columns=[f"label_h{horizon}"])
                self.test_labels_horizons = pd.DataFrame(test_labels, columns=[f"label_h{horizon}"])
            else:
                self.train_labels_horizons[f"label_h{horizon}"] = train_labels
                self.val_labels_horizons[f"label_h{horizon}"] = val_labels
                self.test_labels_horizons[f"label_h{horizon}"] = test_labels
        
        self._normalize_data()
        
    def _label_data(self, input: np.ndarray, window_size: int, horizon: int):
        # TODO: Implement labeling logic for the data
        pass

    def _normalize_data(self):
        # TODO: Implement normalization logic for the data
        pass
        
    def _split_days(self) -> list[int]:
        train = int(self.num_trading_days * self.split_rates[0])
        val = int(self.num_trading_days * self.split_rates[1]) + train
        test = int(self.num_trading_days * self.split_rates[2]) + val
        print(f"There are {train} days for training, {val - train} days for validation and {test - val} days for testing")
        return [train, val, test]
        
        
            
            
            
    
    def _save(self, path_to_save: str):
        # np.save(path_to_save + "/train.npy", self.train_set)
        # np.save(path_to_save + "/val.npy", self.val_set)
        # np.save(path_to_save + "/test.npy", self.test_set)
        pass
        
    def _sample(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        dataframe['timestamp'] = pd.to_datetime(dataframe['timestamp'], errors='coerce')
        # fill up missing snapshot with previous snapshot
        dataframe = dataframe \
            .set_index('timestamp') \
            .resample(self.sampling_time.value) \
            .first() \
            .ffill().reset_index()
        return dataframe
            
            