class DataHandler:
    def __init__(self, cache, label_column, feature_column, MAX_TOKEN_LEN):
        self.cache = cache

        # Load train set
        file = f"{self.cache}raw/train.parq"
        download("https://drive.switch.ch/index.php/s/mRnuzx4BLpMLqyz/download", file)
        train_df = load_dataframe(file)

        # Load test set
        file = f"{self.cache}raw/test.parq"
        download("https://drive.switch.ch/index.php/s/DKUnZraeGp3EIK3/download", file)
        test_df = load_dataframe(file)

        #Define feature and label column
        self.label_column = label_column
        self.feature_column = feature_column

        #Clean the data
        train_df[feature_column] = train_df[feature_column].map(DataHandler._clean_text)
        test_df[feature_column] = test_df[feature_column].map(DataHandler._clean_text)


        #Save text lengths (after cleaning) into variables (can e.g. be used for plotting later)
        self.dist_text_len_train = train_df[feature_column].map(len)
        self.dist_text_len_test = test_df[feature_column].map(len)

        #Define label <--> index mapping
        self.create_label_mapping(train_df, test_df)

        #Test train split
        train_df, val_df = self.train_val_split(train_df)

        #Feature label split
        self.X_train, self.y_train = self.feature_label_split(train_df)
        self.X_test, self.y_test = self.feature_label_split(test_df)
        self.X_val, self.y_val = self.feature_label_split(val_df)

        #tokenize
        self.MAX_TOKEN_LEN=MAX_TOKEN_LEN
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-german-cased")
        self.X_train = np.array([self._tokenize_text(text) for text in tqdm(self.X_train)])
        self.X_test = np.array([self._tokenize_text(text) for text in tqdm(self.X_test)])
        self.X_val = np.array([self._tokenize_text(text) for text in tqdm(self.X_val)])

        #Get indices of labels
        self.y_train_1hot = self.one_hot_encode(self.y_train)
        self.y_test_1hot = self.one_hot_encode(self.y_test)
        self.y_val_1hot = self.one_hot_encode(self.y_val)

        #
        self.calculate_class_weights()


    @staticmethod
    def _clean_text(text):
        """
            - remove any html tags (< /br> often found)
            - Keep only ASCII + European Chars and whitespace, no digits
            - remove single letter chars
            - convert all whitespaces (tabs etc.) to single wspace
        """
        RE_WSPACE = re.compile(r"\s+", re.IGNORECASE)
        RE_TAGS = re.compile(r"<[^>]+>")
        RE_ASCII = re.compile(r"[^A-Za-zÀ-ž,.!? ]", re.IGNORECASE)
        RE_SINGLECHAR = re.compile(r"\b[A-Za-zÀ-ž,.!?]\b", re.IGNORECASE)

        text = re.sub(RE_TAGS, " ", text)
        text = re.sub(RE_ASCII, " ", text)
        text = re.sub(RE_SINGLECHAR, " ", text)
        text = re.sub(RE_WSPACE, " ", text)
        return text


    def plt_text_len_dist(self):
      # Create a figure and a set of subplots
      fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns

      # Histogram for data.dist_text_len_train
      axs[0].hist(self.dist_text_len_train, bins=50, rwidth=0.8)
      axs[0].set_title('Train Set')
      axs[0].set_xlabel('Text Length')
      axs[0].set_ylabel('Frequency')

      # Histogram for data.dist_text_len_test
      axs[1].hist(self.dist_text_len_test, bins=50, rwidth=0.8)
      axs[1].set_title('Test Set')
      axs[1].set_xlabel('Text Length')
      axs[1].set_ylabel('Frequency')

      # Set a title for the entire figure
      plt.suptitle('Text Lengths')

      # Adjust layout for better display
      plt.tight_layout(rect=[0, 0.03, 1, 0.95])

      # Show the plot
      plt.show()

    def _tokenize_text(self, text):
        encoded = self.tokenizer.encode_plus(
        text=text,
        add_special_tokens=True,  # Add `[CLS]` and `[SEP]`
        max_length=self.MAX_TOKEN_LEN,  # Max length to truncate/pad
        padding='max_length',  # Pad sentence to max length
        return_attention_mask=False,  # attention mask not needed for our task
        return_token_type_ids=False,
        truncation=True, )
        return encoded['input_ids']

    def train_val_split(self, train_df, test_size=0.2, random_state=42):
        return train_test_split(train_df,
                                test_size=test_size,
                                random_state=random_state,
                                shuffle=True,
                                stratify=train_df[self.label_column])

    def feature_label_split(self, data):
      X, y = data[self.feature_column], data[self.label_column]
      return  X, y


    def create_label_mapping(self, *data):
        data_all = pd.concat(data)
        self.label_to_index = {label: index for index, label in enumerate(data_all[self.label_column].unique())}
        self.index_to_label = {index: label for label, index in self.label_to_index.items()}

    def one_hot_encode(self, labels):
        return tf.keras.utils.to_categorical([self.label_to_index[label] for label in labels], num_classes=len(self.label_to_index))

    def create_batched_datasets(self, EPOCHS, BATCH_SIZE, batches_in_train = -1):

        #Only use a part of the data for the training in case batches_in_train > 0
        if(batches_in_train > 0):
          X_train = self.X_train[:batches_in_train*BATCH_SIZE] # 4*32 = 128
          y_train_1hot = self.y_train_1hot[:batches_in_train*BATCH_SIZE]
        else:
           X_train = self.X_train
           y_train_1hot = self.y_train_1hot

        self.train_len = len(X_train)

        self.train_dataset = (tf.data.Dataset.from_tensor_slices((X_train, y_train_1hot))
                    .shuffle(buffer_size=len(X_train), reshuffle_each_iteration=True)
                    .repeat(EPOCHS)
                    .batch(BATCH_SIZE))
        self.val_dataset = (tf.data.Dataset.from_tensor_slices((self.X_val, self.y_val_1hot))
                    .batch(BATCH_SIZE))

    def predictions_to_labels(self, predictions, return_labels=True):
        """
        Converts model predictions to either text class labels or their indices.

        Parameters:
        predictions (array-like): The output array from model prediction.
        return_labels (bool): If True, returns text labels; otherwise returns indices.

        Returns:
        An array of text labels or indices based on the predictions.
        """
        # Get the index of the maximum value in each prediction vector (highest probability)
        predicted_indices = np.argmax(predictions, axis=1)

        if return_labels:
            # Convert indices to text labels
            return [self.index_to_label[idx] for idx in predicted_indices]
        else:
            # Return indices
            return predicted_indices

    def calculate_class_weights(self):
        # Convert one-hot encoded labels back to original class labels
        original_labels = np.argmax(self.y_train_1hot, axis=1)

        # Compute class weights
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(original_labels),
            y=original_labels
        )

        # Map the computed weights to the corresponding class indices
        self.class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}

    def save_class_instance(self, name='preprocessed_data'):
        # Only save the data that are not tensorflow objects
        # First, temporarily remove TensorFlow objects
        if hasattr(self, 'train_dataset') and hasattr(self, 'val_dataset'):
          train_dataset_tmp, val_dataset_tmp = self.train_dataset, self.val_dataset
        self.train_dataset, self.val_dataset = None, None

        try:
            with open(os.path.join(self.cache, f'{name}.pkl'), 'wb') as file:
                pickle.dump(self, file)
            #print("Class instance saved successfully.")
        except Exception as e:
            print(f"Error saving class instance: {e}")

        # Restore the TensorFlow objects
        if 'train_dataset_tmp' in locals() and 'val_dataset_tmp' in locals():
          self.train_dataset, self.val_dataset = train_dataset_tmp, val_dataset_tmp

    @staticmethod
    def load_class_instance(path='data/raw/', name='preprocessed_data'):
        # Load the class instance using pickle
        try:
            with open(os.path.join(path, f'{name}.pkl'), 'rb') as file:
                loaded_data = pickle.load(file)
            #print("Class instance loaded successfully.")
        except Exception as e:
            print(f"Error loading class instance: {e}")
            return None
        return loaded_data

