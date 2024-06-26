# حيرة النماذج ذات الطول الثابت

حيرة (PPL) هي واحدة من أكثر المقاييس شيوعًا لتقييم نماذج اللغة. قبل الغوص في الموضوع، يجب أن نلاحظ أن المقياس ينطبق بشكل خاص على نماذج اللغة الكلاسيكية (يُطلق عليها أحيانًا نماذج اللغة الذاتية التعزيز أو السببية) وهي غير محددة جيدًا لنماذج اللغة المقنعة مثل BERT (راجع [ملخص النماذج](model_summary)).

تُعرَّف الحيرة على أنها الأس الأساسي للمتوسط اللوغاريتمي الاحتمالي السلبي لتسلسل. إذا كان لدينا تسلسل مميز \\(X = (x_0, x_1, \dots, x_t)\\)، فإن حيرة \\(X\\) هي،

$$\text{PPL}(X) = \exp \left\{ {-\frac{1}{t}\sum_i^t \log p_\theta (x_i|x_{<i}) } \right\}$$

حيث \\(\log p_\theta (x_i|x_{<i})\\) هو اللوغاريتم الاحتمالي للرمز i المشروط بالرموز السابقة \\(x_{<i}\\) وفقًا لنموذجنا. ومن الناحية البديهية، يمكن اعتبارها تقييمًا لقدرة النموذج على التنبؤ بشكل موحد بين مجموعة من الرموز المحددة في مجموعة من النصوص. ومن المهم أن نلاحظ أن هذا يعني أن إجراء التمييز بين الرموز له تأثير مباشر على حيرة النموذج، والتي يجب أن تؤخذ دائمًا في الاعتبار عند مقارنة النماذج المختلفة.

وهذا يعادل أيضًا أس الأساس للانتروبيا المشتركة بين البيانات وتنبؤات النموذج. لمزيد من البديهيات حول الحيرة وعلاقتها بـ Bits Per Character (BPC) وضغط البيانات، تحقق من هذه [التدوينة الرائعة على The Gradient](https://thegradient.pub/understanding-evaluation-metrics-for-language-models/).

## حساب PPL مع النماذج ذات الطول الثابت

إذا لم نكن مقيدين بحجم سياق النموذج، فسنقوم بتقييم حيرة النموذج عن طريق تفكيك تسلسل بطريقة ذاتية التعزيز والتعامل الشرطي مع التسلسل الفرعي السابق بالكامل في كل خطوة، كما هو موضح أدناه.

![التفكيك الكامل لتسلسل مع طول سياق غير محدود](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/ppl_full.gif)

ومع ذلك، عند العمل مع النماذج التقريبية، عادة ما يكون لدينا قيد على عدد الرموز التي يمكن للنموذج معالجتها. على سبيل المثال، تحتوي أكبر نسخة من [GPT-2](model_doc/gpt2) على طول ثابت يبلغ 1024 رمزًا، لذلك لا يمكننا حساب \\(p_\theta(x_t|x_{<t})\\) مباشرة عندما \\(t\\) أكبر من 1024.

بدلاً من ذلك، يتم عادةً تقسيم التسلسل إلى تسلسلات فرعية تساوي حجم الإدخال الأقصى للنموذج. إذا كان حجم الإدخال الأقصى للنموذج هو \\(k\\)، فإننا نقوم بعد ذلك بتقريب احتمال الرمز \\(x_t\\) عن طريق التعامل الشرطي فقط مع \\(k-1\\) من الرموز التي تسبقه بدلاً من السياق بأكمله. عند تقييم حيرة النموذج لتسلسل، هناك نهج مغرٍ ولكنه دون المستوى الأمثل وهو تقسيم التسلسل إلى قطع غير متداخلة وإضافة اللوغاريتميات المفككة لكل جزء بشكل مستقل.

![حيرة غير مثالية لا تستفيد من السياق الكامل المتاح](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/ppl_chunked.gif)

هذا سريع الحساب لأن حيرة كل جزء يمكن حسابها في تمرير واحد للأمام، ولكنه يمثل تقريبًا سيئًا للحيرة المفككة بالكامل وسيؤدي عادةً إلى حيرة أعلى (أسوأ) لأن النموذج سيكون لديه سياق أقل في معظم خطوات التنبؤ.

بدلاً من ذلك، يجب تقييم حيرة النماذج ذات الطول الثابت باستخدام إستراتيجية النافذة المنزلقة. ينطوي هذا على تحريك نافذة السياق بشكل متكرر بحيث يكون للنموذج سياق أكبر عند إجراء كل تنبؤ.

![حيرة النافذة المنزلقة التي تستفيد من كل السياق المتاح](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/ppl_sliding.gif)

هذا تقريب أوثق للتفكيك الحقيقي لاحتمالية التسلسل وعادة ما يؤدي إلى نتيجة أفضل. الجانب السلبي هو أنه يتطلب تمريرًا للأمام لكل رمز في المجموعة. حل وسط عملي جيد هو استخدام نافذة منزلقة ذات خطوة، حيث تتحرك النافذة بخطوات أكبر بدلاً من الانزلاق بمقدار 1 رمز في كل مرة. يسمح ذلك بإجراء الحساب بشكل أسرع مع إعطاء النموذج سياقًا كبيرًا للتنبؤات في كل خطوة.

## مثال: حساب الحيرة مع GPT-2 في 🤗 Transformers

دعونا نوضح هذه العملية مع GPT-2.

```python
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

device = "cuda"
model_id = "openai-community/gpt2-large"
model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
```

سنقوم بتحميل مجموعة بيانات WikiText-2 وتقييم الحيرة باستخدام بعض إستراتيجيات النافذة المنزلقة المختلفة. نظرًا لأن هذه المجموعة صغيرة ونقوم بتمرير واحد فقط عبر المجموعة، فيمكننا تحميل المجموعة وتشفيرها بالكامل في الذاكرة.

```python
from datasets import load_dataset

test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")
```

مع 🤗 Transformers، يمكننا ببساطة تمرير `input_ids` كـ `labels` إلى نموذجنا، ويتم إرجاع متوسط اللوغاريتميات السلبية لكل رمز كخسارة. ومع ذلك، مع نهج النافذة المنزلقة لدينا، هناك تداخل في الرموز التي نمررها إلى النموذج في كل تكرار. لا نريد أن يتم تضمين اللوغاريتميات الاحتمالية للرموز التي نتعامل معها فقط كسياق في خسارتنا، لذا يمكننا تعيين هذه الأهداف إلى `-100` بحيث يتم تجاهلها. ما يلي هو مثال على كيفية القيام بذلك مع خطوة من `512`. وهذا يعني أن النموذج سيكون لديه 512 رمزًا على الأقل كسياق عند حساب الاحتمالية الشرطية لأي رمز واحد (شريطة أن تكون هناك 512 رمزًا سابقًا متاحًا للتعامل الشرطي معه).

```python
import torch
from tqdm import tqdm

max_length = model.config.n_positions
stride = 512
seq_len = encodings.input_ids.size(1)

nlls = []
prev_end_loc = 0
for begin_loc in tqdm(range(0, seq_len, stride)):
end_loc = min(begin_loc + max_length, seq_len)
trg_len = end_loc - prev_end_loc  # قد يختلف عن الخطوة في الحلقة الأخيرة
input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
target_ids = input_ids.clone()
target_ids[:, :-trg_len] = -100

with torch.no_grad():
outputs = model(input_ids, labels=target_ids)

# يتم حساب الخسارة باستخدام CrossEntropyLoss الذي يحسب المتوسط على التصنيفات الصالحة
# لاحظ أن النموذج يحسب الخسارة فقط على trg_len - 1 من التصنيفات، لأنه يتحول داخليًا إلى اليسار بواسطة 1.
neg_log_likelihood = outputs.loss

nlls.append(neg_log_likelihood)

prev_end_loc = end_loc
if end_loc == seq_len:
break

ppl = torch.exp(torch.stack(nlls).mean())
```

يعطي تشغيل هذا مع طول الخطوة يساوي طول الإدخال الأقصى نتيجة مماثلة لاستراتيجية النافذة غير المنزلقة التي ناقشناها أعلاه. كلما صغرت الخطوة، زاد السياق الذي سيكون لدى النموذج في إجراء كل تنبؤ، وكلما كانت الحيرة المبلغ عنها أفضل عادةً.

عندما نقوم بتشغيل ما سبق باستخدام `stride = 1024`، أي بدون تداخل، تكون نتيجة PPL هي `19.44`، وهو ما يماثل `19.93` المبلغ عنه في ورقة GPT-2. من خلال استخدام `stride = 512` وبالتالي استخدام إستراتيجية النافذة المنزلقة لدينا، ينخفض هذا إلى `16.45`. هذه النتيجة ليست فقط أفضل، ولكنها محسوبة بطريقة أقرب إلى التفكيك الذاتي التعزيز الحقيقي لاحتمالية التسلسل.