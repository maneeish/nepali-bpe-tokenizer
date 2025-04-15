from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from tokenizers.processors import BertProcessing
from tokenizers.normalizers import NFKC, Lowercase, Sequence
import os

# Step 1: Initialize empty BPE model
tokenizer = Tokenizer(models.BPE())

# Step 2: Normalization and Pre-tokenization
tokenizer.normalizer = Sequence([NFKC(), Lowercase()])
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

# Step 3: Trainer
trainer = trainers.BpeTrainer(vocab_size=8000, show_progress=True, special_tokens=[
    "<pad>", "<unk>", "<s>", "</s>", "<mask>"
])

# Step 4: Load your Nepali text files
files = ["nepali_text.txt"]


tokenizer.train(files, trainer)

# Step 5: Save tokenizer
tokenizer.save("nepali-bpe-tokenizer.json")

print("Tokenizer trained and saved!")


# To test the code
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("nepali-bpe-tokenizer.json")
output = tokenizer.encode("भर्ना अभियान, पठनपाठन र परीक्षा बहिष्कार गर्ने शिक्षकको घोषणा")

print("Tokens:", output.tokens)
print("IDs:", output.ids)
