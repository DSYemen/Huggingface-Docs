# GGUF وتفاعلها مع المحولات 

يستخدم تنسيق ملف GGUF لتخزين النماذج للاستدلال باستخدام [GGML](https://github.com/ggerganov/ggml) والمكتبات الأخرى التي تعتمد عليه، مثل [llama.cpp](https://github.com/ggerganov/llama.cpp) أو [whisper.cpp](https://github.com/ggerganov/whisper.cpp) الشهيرة جدًا. 

وهو تنسيق ملف [مدعوم من قبل Hugging Face Hub](https://huggingface.co/docs/hub/en/gguf) بميزات تسمح بفحص سريع للتوابع والبيانات الوصفية داخل الملف. 

تم تصميم تنسيق الملف هذا كـ"تنسيق ملف واحد" حيث يحتوي ملف واحد عادةً على كل من سمات التكوين، ومعجم محدد الرموز، والسمات الأخرى، بالإضافة إلى جميع التوابع التي سيتم تحميلها في النموذج. وتأتي هذه الملفات بتنسيقات مختلفة وفقًا لنوع التكميم في الملف. نلقي نظرة موجزة على بعضها [هنا](https://huggingface.co/docs/hub/en/gguf#quantization-types). 

## الدعم داخل المحولات 

لقد أضفنا القدرة على تحميل ملفات `gguf` داخل `المحولات` لتوفير قدرات تدريب/ضبط دقيق إضافية لنماذج gguf، قبل إعادة تحويل تلك النماذج إلى `gguf` لاستخدامها داخل نظام بيئي `ggml`. عند تحميل نموذج، نقوم أولاً بإلغاء تكميمه إلى fp32، قبل تحميل الأوزان لاستخدامها في PyTorch. 

> [!ملاحظة]
> لا يزال الدعم استكشافيًا للغاية ونرحب بالمساهمات لتعزيزه عبر أنواع التكميم وبنيات النماذج. 

فيما يلي بنيات النماذج المدعومة وأنواع التكميم: 

### أنواع التكميم المدعومة 

تتحدد أنواع التكميم المدعومة الأولية وفقًا لملفات التكميم الشائعة التي تمت مشاركتها على Hub. 

- F32
- Q2_K
- Q3_K
- Q4_0
- Q4_K
- Q5_K
- Q6_K
- Q8_0 

نأخذ مثالاً من محلل Python الممتاز [99991/pygguf](https://github.com/99991/pygguf) لإلغاء تكميم الأوزان. 

### بنيات النماذج المدعومة 

في الوقت الحالي، فإن بنيات النماذج المدعومة هي البنيات التي كانت شائعة جدًا على Hub، وهي: 

- LLaMa
- Mistral
- Qwen2 

## مثال الاستخدام 

لتحميل ملفات `gguf` في `المحولات`، يجب تحديد وسيط `gguf_file` لأساليب `from_pretrained` لكل من محددات الرموز والنماذج. فيما يلي كيفية تحميل محدد الرموز والنموذج، والتي يمكن تحميلها من نفس الملف بالضبط: 

```py
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
filename = "tinyllama-1.1b-chat-v1.0.Q6_K.gguf"

tokenizer = AutoTokenizer.from_pretrained(model_id, gguf_file=filename)
model = AutoModelForCausalLM.from_pretrained(model_id, gguf_file=filename)
``` 

الآن لديك حق الوصول إلى الإصدار الكامل غير المُكمم للنموذج في نظام PyTorch البيئي، حيث يمكنك دمجه مع مجموعة من الأدوات الأخرى. 

لإعادة التحويل إلى ملف `gguf`، نوصي باستخدام ملف [`convert-hf-to-gguf.py`](https://github.com/ggerganov/llama.cpp/blob/master/convert-hf-to-gguf.py) من llama.cpp. 

فيما يلي كيفية استكمال البرنامج النصي أعلاه لحفظ النموذج وتصديره مرة أخرى إلى `gguf`: 

```py
tokenizer.save_pretrained('directory')
model.save_pretrained('directory')

!python ${path_to_llama_cpp}/convert-hf-to-gguf.py ${directory}
```