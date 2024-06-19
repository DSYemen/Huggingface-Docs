# Quanto

مكتبة 🤗Quanto هي مجموعة أدوات كمية متنوعة تعتمد على PyTorch. طريقة الكَمْنَة المستخدمة هي الكَمْنَة الخطية. يوفر Quanto العديد من الميزات الفريدة مثل:

- كَمْنَة الأوزان (`float8`، `int8`، `int4`، `int2`)
- كَمْنَة التنشيط (`float8`، `int8`)
- عدم التخصص في طريقة معينة (مثل الرؤية الحاسوبية، ونماذج اللغة الكبيرة)
- عدم التخصص في جهاز معين (مثل CUDA و MPS و CPU)
- التوافق مع `torch.compile`
- سهولة إضافة نواة مخصصة لجهاز محدد
- دعم التدريب الواعي بالكمية

قبل البدء، تأكد من تثبيت المكتبات التالية:

```bash
pip install quanto accelerate transformers
```

الآن يمكنك كَمْنَة نموذج عن طريق تمرير كائن [`QuantoConfig`] في طريقة [`~PreTrainedModel.from_pretrained`]. يعمل هذا مع أي نموذج في أي طريقة، طالما أنه يحتوي على طبقات `torch.nn.Linear`.

يدعم التكامل مع مكتبة Transformers كَمْنَة الأوزان فقط. بالنسبة لحالات الاستخدام الأكثر تعقيدًا مثل كَمْنَة التنشيط، والمعايرة، والتدريب الواعي بالكمية، يجب استخدام مكتبة [quanto](https://github.com/huggingface/quanto) بدلاً من ذلك.

```py
from transformers import AutoModelForCausalLM, AutoTokenizer, QuantoConfig

model_id = "facebook/opt-125m"
tokenizer = AutoTokenizer.from_pretrained(model_id)
quantization_config = QuantoConfig(weights="int8")
quantized_model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda:0", quantization_config=quantization_config)
```

لاحظ أن التسلسل الهرمي غير مدعوم حتى الآن مع Transformers ولكنه قادم قريبًا! إذا كنت تريد حفظ النموذج، فيمكنك استخدام مكتبة quanto بدلاً من ذلك.

تستخدم مكتبة quanto خوارزمية الكَمْنَة الخطية للكَمْنَة. على الرغم من أن هذه تقنية كَمْنَة أساسية، إلا أننا نحصل على نتائج جيدة جدًا! الق نظرة على المعيار المرجعي التالي (llama-2-7b على مقياس الحيرة). يمكنك العثور على المزيد من المعايير المرجعية [هنا](https://github.com/huggingface/quanto/tree/main/bench/generation)

تتمتع المكتبة بالمرونة الكافية لتكون متوافقة مع معظم خوارزميات تحسين PTQ. وتتمثل الخطة المستقبلية في دمج الخوارزميات الأكثر شعبية بأكثر الطرق سلاسة (AWQ، Smoothquant).