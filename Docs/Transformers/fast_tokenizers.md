# استخدام المحللات من 🤗 Tokenizers

يعتمد [`PreTrainedTokenizerFast`] على مكتبة [🤗 Tokenizers](https://huggingface.co/docs/tokenizers). يمكن تحميل المحللات التي تم الحصول عليها من مكتبة 🤗 Tokenizers ببساطة شديدة في 🤗 Transformers.

قبل الدخول في التفاصيل، دعونا نبدأ أولاً بإنشاء محلل وهمي في بضع سطور:

```python
>>> from tokenizers import Tokenizer
>>> from tokenizers.models import BPE
>>> from tokenizers.trainers import BpeTrainer
>>> from tokenizers.pre_tokenizers import Whitespace

>>> tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
>>> trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

>>> tokenizer.pre_tokenizer = Whitespace()
>>> files = [...]
>>> tokenizer.train(files, trainer)
```

الآن لدينا محلل مدرب على الملفات التي حددناها. يمكننا إما الاستمرار في استخدامه في وقت التشغيل هذا، أو حفظه في ملف JSON لإعادة استخدامه في المستقبل.

## التحميل مباشرة من كائن المحلل

دعونا نرى كيف يمكننا الاستفادة من كائن المحلل هذا في مكتبة 🤗 Transformers. تسمح فئة [`PreTrainedTokenizerFast`] بالتشغيل الفوري، من خلال قبول كائن *tokenizer* الذي تم إنشاؤه كوسيط:

```python
>>> from transformers import PreTrainedTokenizerFast

>>> fast_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
```

يمكن الآن استخدام هذا الكائن مع جميع الطرق التي تشترك فيها محولات 🤗! انتقل إلى [صفحة المحلل](main_classes/tokenizer) لمزيد من المعلومات.

## التحميل من ملف JSON

لحميل محلل من ملف JSON، دعونا نبدأ أولاً بحفظ محللنا:

```python
>>> tokenizer.save("tokenizer.json")
```

يمكن تمرير المسار الذي حفظنا فيه هذا الملف إلى طريقة تهيئة [`PreTrainedTokenizerFast`] باستخدام معلمة `tokenizer_file`:

```python
>>> from transformers import PreTrainedTokenizerFast

>>> fast_tokenizer = PreTrainedTokenizerFast(tokenizer_file="tokenizer.json")
```

يمكن الآن استخدام هذا الكائن مع جميع الطرق التي تشترك فيها محولات 🤗! انتقل إلى [صفحة المحلل](main_classes/tokenizer) لمزيد من المعلومات.