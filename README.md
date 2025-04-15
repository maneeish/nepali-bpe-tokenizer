# Nepali BPE Tokenizer

A custom Byte-Pair Encoding (BPE) tokenizer for the Nepali language, built using the Hugging Face `tokenizers` library. This tokenizer is trained from scratch on Nepali text data and designed to support downstream NLP tasks like classification, translation, and chatbots.

---

## 🚀 Features

- Byte-Pair Encoding (BPE) model
- Unicode normalization (NFKC + Lowercase)
- Whitespace pre-tokenization
- Trained on Nepali text data
- Special tokens support: `<pad>`, `<unk>`, `<s>`, `</s>`, `<mask>`
- Outputs subword-level tokens and token IDs
- Saves and loads from a `.json` file

---

 ## 🛠 Installation

```bash
pip install tokenizers
```

---

## 📦 Training the Tokenizer

```python
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from tokenizers.processors import BertProcessing
from tokenizers.normalizers import NFKC, Lowercase, Sequence

# Initialize empty BPE model
tokenizer = Tokenizer(models.BPE())

# Normalization and Pre-tokenization
tokenizer.normalizer = Sequence([NFKC(), Lowercase()])
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

# Trainer
trainer = trainers.BpeTrainer(
    vocab_size=8000,
    show_progress=True,
    special_tokens=["<pad>", "<unk>", "<s>", "</s>", "<mask>"]
)

# Load Nepali text file(s)
files = ["nepali_text.txt"]
tokenizer.train(files, trainer)

# Save tokenizer
tokenizer.save("nepali-bpe-tokenizer.json")
print("Tokenizer trained and saved!")
```

---

## 🔍 Testing the Tokenizer

```python
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("nepali-bpe-tokenizer.json")
output = tokenizer.encode("भर्ना अभियान, पठनपाठन र परीक्षा बहिष्कार गर्ने शिक्षकको घोषणा")

print("Tokens:", output.tokens)
print("IDs:", output.ids)
```

---

## 📄 Example Output

**Input:**
```
"भर्ना अभियान, पठनपाठन र परीक्षा बहिष्कार गर्ने शिक्षकको घोषणा"
```

**Tokens:**
```
['भर्ना', 'अभियान', ',', 'पठनपाठन', 'र', 'परीक्षा', 'बहिष्कार', 'गर्ने', 'शिक्षकको', 'घोषणा']
```

**Token IDs:**
```
[42, 81, 89, 12, 42, 127, 100, 7, 408, 38, 126, 30, 38, 45, 623, 54, 473, 41, 270, 98, 118, 352, 234, 138, 19, 82, 22, 60, 92, 52]
```

---

## 📚 Dataset

Ensure that `nepali_text.txt` contains clean and representative Nepali language sentences.

---

## 🤝 Contributing

Pull requests are welcome. If you'd like to contribute or train on a larger dataset, feel free to fork and open a PR!

---

## 📜 License

MIT License

---

## ✨ Acknowledgements

- [Hugging Face Tokenizers](https://github.com/huggingface/tokenizers)
- Nepali NLP community

---

## 💡 Future Work

- Add sentencepiece tokenizer version
- Train on a larger Nepali corpus
- Create Hugging Face-compatible tokenizer class
- Integrate with transformers pipeline

---

## 🔗 Connect with Me

**Manish Mandal**  
📧 Gmail : maneeish09@gmail.com  
🔗 LinkedIn : https://www.linkedin.com/in/manish-mandal-6b7212295/


