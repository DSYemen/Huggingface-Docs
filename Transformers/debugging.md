# استكشاف الأخطاء وإصلاحها

يمكن أن يكون التدريب على وحدات معالجة الرسومات (GPU) المتعددة مهمة صعبة، سواء كنت تواجه مشكلات في التثبيت أو مشكلات في الاتصال بين وحدات معالجة الرسومات الخاصة بك. يغطي دليل استكشاف الأخطاء وإصلاحها هذا بعض المشكلات التي قد تواجهها وكيفية حلها.

## تثبيت CUDA في DeepSpeed

إذا كنت تستخدم DeepSpeed، فمن المحتمل أنك قمت بتثبيته بالفعل باستخدام الأمر التالي.

```bash
pip install deepspeed
```

يقوم DeepSpeed بتجميع كود CUDA C++، ويمكن أن يكون مصدرًا محتملًا للأخطاء عند بناء امتدادات PyTorch التي تتطلب CUDA. تعتمد هذه الأخطاء على كيفية تثبيت CUDA على نظامك، ويركز هذا القسم على PyTorch المبني باستخدام CUDA 10.2.

### حزم أدوات CUDA غير المتطابقة

يأتي PyTorch مع حزمة أدوات CUDA الخاصة به، ولكن لاستخدام DeepSpeed مع PyTorch، تحتاج إلى وجود إصدار متطابق من CUDA مثبت على مستوى النظام. على سبيل المثال، إذا قمت بتثبيت PyTorch مع `cudatoolkit==10.2` في بيئة Python الخاصة بك، فستحتاج أيضًا إلى تثبيت CUDA 10.2 على مستوى النظام. إذا لم يكن لديك CUDA مثبتًا على مستوى النظام، فيجب تثبيته أولاً.

قد يختلف الموقع الدقيق من نظام إلى آخر، ولكن `usr/local/cuda-10.2` هو الموقع الأكثر شيوعًا على العديد من أنظمة Unix. عندما يتم إعداد CUDA بشكل صحيح وإضافته إلى متغير البيئة `PATH` الخاص بك، يمكنك العثور على موقع التثبيت باستخدام الأمر التالي:

```bash
which nvcc
```

### حزم أدوات CUDA متعددة

قد يكون لديك أيضًا أكثر من حزمة أدوات CUDA واحدة مثبتة على مستوى النظام.

```bash
/usr/local/cuda-10.2
/usr/local/cuda-11.0
```

عادةً، يقوم مثبت الحزمة بتعيين المسارات إلى الإصدار الأخير الذي تم تثبيته. إذا فشلت حزمة البناء لأنها لا يمكنها العثور على الإصدار الصحيح من CUDA (على الرغم من تثبيته بالفعل على مستوى النظام)، فيجب عليك تكوين متغيرات البيئة `PATH` و`LD_LIBRARY_PATH` للإشارة إلى المسار الصحيح.

الق نظرة على محتويات متغيرات البيئة هذه أولاً:

```bash
echo $PATH
echo $LD_LIBRARY_PATH
```

يسرد `PATH` مواقع الملفات القابلة للتنفيذ، بينما يسرد `LD_LIBRARY_PATH` أين تبحث عن المكتبات المشتركة. يتم منح الأولوية للمدخلات السابقة على اللاحقة، ويتم استخدام `:` للفصل بين الإدخالات المتعددة. لإخبار برنامج البناء بمكان العثور على حزمة أدوات CUDA المحددة التي تريدها، أدخل المسار الصحيح في القائمة أولاً. يقوم هذا الأمر بإلحاق المسار بالقيم الموجودة بدلاً من الكتابة فوقها.

```bash
# ضبط الإصدار والمسار الكامل حسب الحاجة
تصدير المسار =/ usr/local/cuda-10.2/bin:$PATH
تصدير LD_LIBRARY_PATH =/ usr/local/cuda-10.2/lib64: $LD_LIBRARY_PATH
```

بالإضافة إلى ذلك، يجب عليك أيضًا التحقق من وجود المجلدات التي تقوم بتعيينها بالفعل. يحتوي المجلد الفرعي `lib64` على كائنات CUDA `.so` المختلفة (مثل `libcudart.so`) وعلى الرغم من أنه من غير المحتمل أن يقوم نظامك بتسميتها بشكل مختلف، يجب عليك التحقق من الأسماء الفعلية وتغييرها وفقًا لذلك.

### إصدارات CUDA الأقدم

في بعض الأحيان، قد ترفض إصدارات CUDA الأقدم البناء مع برامج التجميع الأحدث. على سبيل المثال، إذا كان لديك `gcc-9` ولكن CUDA يريد `gcc-7`. عادةً، يمكّن تثبيت حزمة أدوات CUDA الأحدث الدعم لمجمع أحدث.

يمكنك أيضًا تثبيت إصدار أقدم من المجمع بالإضافة إلى الإصدار الذي تستخدمه حاليًا (أو قد يكون مثبتًا بالفعل ولكنه غير مستخدم بشكل افتراضي ولا يمكن لنظام البناء رؤيته). لحل هذا، يمكنك إنشاء رابط رمزي لمنح نظام البناء إمكانية رؤية المجمع الأقدم.

```bash
# تكييف المسار مع نظامك
sudo ln -s /usr/bin/gcc-7 /usr/local/cuda-10.2/bin/gcc
sudo ln -s /usr/bin/g++-7 /usr/local/cuda-10.2/bin/g++
```

### البناء المسبق

إذا كنت لا تزال تواجه مشكلات في تثبيت DeepSpeed أو إذا كنت تقوم ببناء DeepSpeed في وقت التشغيل، فيمكنك محاولة البناء المسبق لوحدات DeepSpeed قبل تثبيتها. لإجراء بناء محلي لـ DeepSpeed:

```bash
git clone https://github.com/microsoft/DeepSpeed/
cd DeepSpeed
rm -rf build
TORCH_CUDA_ARCH_LIST="8.6" DS_BUILD_CPU_ADAM=1 DS_BUILD_UTILS=1 pip install . \
--global-option="build_ext" --global-option="-j8" --no-cache -v \
--disable-pip-version-check 2>&1 | tee build.log
```

لاستخدام NVMe offload، أضف معلمة `DS_BUILD_AIO=1` إلى أمر البناء وتأكد من تثبيت حزمة libaio-dev على مستوى النظام.

بعد ذلك، سيتعين عليك تحديد بنية GPU الخاصة بك عن طريق تحرير متغير `TORCH_CUDA_ARCH_LIST` (ابحث عن قائمة كاملة من وحدات معالجة الرسوميات من NVIDIA وبنيتها المقابلة على هذه [الصفحة](https://developer.nvidia.com/cuda-gpus)). للتحقق من إصدار PyTorch الذي يتوافق مع بنيتك، قم بتشغيل الأمر التالي:

```bash
python -c "import torch; print(torch.cuda.get_arch_list())"
```

للعثور على بنية GPU، استخدم الأمر التالي:

`?????????????????????????????????`
## اكتشاف نقص التدفق وطفحان 
إذا بدأت في الحصول على "loss=NaN" أو أعاق النموذج سلوكاً غير طبيعي آخر بسبب "inf" أو "nan" في التنشيطات أو الأوزان، فمن الضروري اكتشاف المكان الذي يحدث فيه أول نقص في التدفق أو طفحان وما الذي أدى إليه. لحسن الحظ، يمكنك تحقيق ذلك بسهولة عن طريق تنشيط وحدة نمطية خاصة ستقوم بالكشف التلقائي.

إذا كنت تستخدم ["Trainer"]، فما عليك سوى إضافة:
```bash
--debug underflow_overflow
```
إلى وسيطات سطر الأوامر العادية، أو تمرير "debug="underflow_overflow" عند إنشاء كائن ["TrainingArguments"].

إذا كنت تستخدم حلقة تدريب خاصة بك أو "Trainer" آخر، فيمكنك تحقيق نفس الشيء بما يلي:
```python
from transformers.debug_utils import DebugUnderflowOverflow

debug_overflow = DebugUnderflowOverflow(model)
```
تُدخل ["~debug_utils.DebugUnderflowOverflow"] خطافات في النموذج الذي سيختبر على الفور بعد كل مكالمة للأمام متغيرات الإدخال والإخراج وكذلك أوزان الوحدة النمطية المقابلة. بمجرد اكتشاف "inf" أو "nan" في عنصر واحد على الأقل من التنشيطات أو الأوزان، سيؤكد البرنامج ويطبع تقريراً مثل هذا (تم اكتشافه باستخدام "google/mt5-small" في الدقة العائمة المختلطة):
```
Detected inf/nan during batch_number=0
Last 21 forward frames:
abs min  abs max  metadata
encoder.block.1.layer.1.DenseReluDense.dropout Dropout
0.00e+00 2.57e+02 input[0]
0.00e+00 2.85e+02 output
[...]
encoder.block.2.layer.0 T5LayerSelfAttention
6.78e-04 3.15e+03 input[0]
2.65e-04 3.42e+03 output[0]
None output[1]
2.25e-01 1.00e+04 output[2]
encoder.block.2.layer.1.layer_norm T5LayerNorm
8.69e-02 4.18e-01 weight
2.65e-04 3.42e+03 input[0]
1.79e-06 4.65e+00 output
encoder.block.2.layer.1.DenseReluDense.wi_0 Linear
2.17e-07 4.50e+00 weight
1.79e-06 4.65e+00 input[0]
2.68e-06 3.70e+1 output
encoder.block.2.layer.1.DenseReluDense.wi_1 Linear
8.08e-07 2.66e+1 weight
1.79e-06 4.65e+00 input[0]
1.27e-04 2.37e+2 output
encoder.block.2.layer.1.DenseReluDense.dropout Dropout
0.00e+00 8.76e+03 input[0]
0.00e+00 9.74e+03 output
encoder.block.2.layer.1.DenseReluDense.wo Linear
1.01e-06 6.44e+00 weight
0.00e+00 9.74e+03 input[0]
3.18e-04 6.27e+04 output
encoder.block.2.layer.1.DenseReluDense T5DenseGatedGeluDense
1.79e-06 4.65e+00 input[0]
3.18e-04 6.27e+04 output
encoder.block.2.layer.1.dropout Dropout
3.18e-04 6.27e+04 input[0]
0.00e+00      inf output
```
تم تقليم مثال الإخراج في المنتصف للإيجاز.

يُظهر العمود الثاني قيمة أكبر عنصر مطلق، لذا إذا نظرت عن كثب إلى الإطارات القليلة الأخيرة، فستجد أن الإدخالات والمخرجات كانت في نطاق "1e4". لذا عندما تم إجراء هذا التدريب في الدقة العائمة المختلطة "fp16"، حدث طفحان في الخطوة الأخيرة (نظرًا لأن أكبر رقم قبل "inf" في "fp16" هو "64e3"). لتجنب حدوث طفحان في "fp16"، يجب أن تظل التنشيطات أقل بكثير من "1e4"، لأن "1e4 * 1e4 = 1e8" لذا فإن أي عملية ضرب المصفوفة مع تنشيطات كبيرة سوف تؤدي إلى حالة طفحان رقمي.

في بداية التتبع، يمكنك اكتشاف رقم الدفعة التي حدثت فيها المشكلة (هنا "Detected inf/nan during batch_number=0" يعني أن المشكلة حدثت في الدفعة الأولى).

يبدأ كل إطار تم الإبلاغ عنه بالإعلان عن الإدخال الكامل المؤهل للوحدة النمطية المقابلة التي يبلغ عنها هذا الإطار. إذا نظرنا فقط إلى هذا الإطار:
```
encoder.block.2.layer.1.layer_norm T5LayerNorm
8.69e-02 4.18e-01 weight
2.65e-04 3.42e+03 input[0]
1.79e-06 4.65e+00 output
```
هنا، يشير "encoder.block.2.layer.1.layer_norm" إلى أنه كان طبقة التطبيع للطبقة الأولى، من الكتلة الثانية للترميز. والمكالمات المحددة لـ "forward" هي "T5LayerNorm".

لنلقِ نظرة على الإطارات القليلة الأخيرة من هذا التقرير:
```
Detected inf/nan during batch_number=0
Last 21 forward frames:
abs min  abs max  metadata
[...]
encoder.block.2.layer.1.DenseReluDense.wi_0 Linear
2.17e-07 4.50e+00 weight
1.79e-06 4.65e+00 input[0]
2.68e-06 3.70e+1 output
encoder.block.2.layer.1.DenseReluDense.wi_1 Linear
8.08e-07 2.66e+1 weight
1.79e-06 4.65e+00 input[0]
1.27e-04 2.37e+2 output
encoder.block.2.layer.1.DenseReluDense.wo Linear
1.01e-06 6.44e+00 weight
0.00e+00 9.74e+03 input[0]
3.18e-04 6.27e+04 output
encoder.block.2.layer.1.DenseReluDense T5DenseGatedGeluDense
1.79e-06 4.65e+00 input[0]
3.18e-04 6.27e+04 output
encoder.block.2.layer.1.dropout Dropout
3.18e-04 6.27e+04 input[0]
0.00e+00      inf output
```
يبلغ الإطار الأخير عن "Dropout.forward" مع الإدخال الوحيد للإدخال الأول والإخراج الثاني للإخراج الوحيد. يمكنك أن ترى أنه تم استدعاؤه من سمة "dropout" داخل فئة "DenseReluDense". يمكننا أن نرى أن هذا حدث أثناء الطبقة الأولى، من الكتلة الثانية، أثناء الدفعة الأولى. وأخيرًا، كان أكبر إدخال مطلق هو "6.27e+04" ونفس الإخراج كان "inf".

يمكنك أن ترى هنا أن "T5DenseGatedGeluDense.forward" أسفر عن تنشيطات إخراج، كان الحد الأقصى المطلق لقيمتها حوالي 62.7 ألف، وهو قريب جدًا من الحد الأعلى لـ "fp16" وهو 64 ألف. في الإطار التالي، لدينا "Dropout" الذي يعيد تطبيع الأوزان، بعد أن قام بإلغاء تنشيط بعض العناصر، مما يدفع القيمة القصوى المطلقة إلى أكثر من 64 ألف، ونحصل على طفحان ("inf").

كما ترى، فإن الإطارات السابقة هي التي نحتاج إلى النظر فيها عندما تبدأ الأرقام في الارتفاع إلى أرقام كبيرة جدًا لـ "fp16".

دعنا نطابق التقرير مع الكود من "models/t5/modeling_t5.py":
```python
class T5DenseGatedGeluDense(nn.Module):
def __init__(self, config):
super().__init__()
self.wi_0 = nn.Linear(config.d_model, config.d_ff, bias=False)
self.wi_1 = nn.Linear(config.d_model, config.d_ff, bias=False)
self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
self.dropout = nn.Dropout(config.dropout_rate)
self.gelu_act = ACT2FN["gelu_new"]

def forward(self, hidden_states):
hidden_gelu = self.gelu_act(self.wi_0(hidden_states))
hidden_linear = self.wi_1(hidden_states)
hidden_states = hidden_gelu * hidden_linear
hidden_states = self.dropout(hidden_states)
hidden_states = self.wo(hidden_states)
return hidden_states
```
الآن من السهل رؤية مكالمة "dropout"، وجميع المكالمات السابقة أيضًا.

نظرًا لأن الكشف يحدث في خطاف للأمام، يتم طباعة هذه التقارير على الفور بعد عودة كل "forward".

بالعودة إلى التقرير الكامل، للتصرف بناءً عليه وإصلاح المشكلة، نحتاج إلى الانتقال إلى أعلى قليلًا من الإطارات حيث بدأت الأرقام في الارتفاع ومن المحتمل التبديل إلى وضع "fp32" هنا، بحيث لا تطفح الأرقام عند ضربها أو جمعها. بالطبع، قد تكون هناك حلول أخرى. على سبيل المثال، يمكننا إيقاف تشغيل "amp" مؤقتًا إذا تم تمكينه، بعد نقل "forward" الأصلي إلى معين مساعد، مثل هذا:
```python
def _forward(self, hidden_states):
hidden_gelu = self.gelu_act(self.wi_0(hidden_states))
hidden_linear = self.wi_1(hidden_states)
hidden_states = hidden_gelu * hidden_linear
hidden_states = self.dropout(hidden_states)
hidden_states = self.wo(hidden_states)
return hidden_states


import torch


def forward(self, hidden_states):
if torch.is_autocast_enabled():
with torch.cuda.amp.autocast(enabled=False):
return self._forward(hidden_states)
else:
return self._forward(hidden_states)
```
نظرًا لأن الكاشف التلقائي يبلغ فقط عن إدخالات ومخرجات الإطارات الكاملة، بمجرد معرفة المكان الذي تبحث فيه، فقد ترغب في تحليل المراحل الوسيطة لأي دالة "forward" محددة أيضًا. في مثل هذه الحالة، يمكنك استخدام دالة المساعدة "detect_overflow" لإدخال الكاشف حيث تريده، على سبيل المثال:
```python
from debug_utils import detect_overflow


class T5LayerFF(nn.Module):
[...]

def forward(self, hidden_states):
forwarded_states = self.layer_norm(hidden_states)
detect_overflow(forwarded_states, "after layer_norm")
forwarded_states = self.DenseReluDense(forwarded_states)
detect_overflow(forwarded_states, "after DenseReluDense")
return hidden_states + self.dropout(forwarded_states)
```
يمكنك أن ترى أننا أضفنا اثنين من هذه العناصر، والآن نقوم بتتبع ما إذا تم اكتشاف "inf" أو "nan" لـ "forwarded_states" في مكان ما بينهما.

في الواقع، يقوم الكاشف بالفعل بالإبلاغ عن هذه العناصر لأن كل مكالمة في المثال أعلاه هي "nn.Module"، ولكن لنفترض أنك أجريت بعض الحسابات المباشرة المحلية، فهذا هو ما ستفعله.

بالإضافة إلى ذلك، إذا كنت تقوم بتنشيط الكاشف في الكود الخاص بك، فيمكنك ضبط عدد الإطارات المطبوعة من الإعداد الافتراضي، على سبيل المثال:
```python
from transformers.debug_utils import DebugUnderflowOverflow

debug_overflow = DebugUnderflowOverflow(model, max_frames_to_save=100)
```
### تتبع القيمة المطلقة الدنيا والقصوى الدفعية المحددة

يمكن استخدام نفس فئة التصحيح لتتبع الدفعة مع إيقاف تشغيل ميزة اكتشاف التدفق تحت أو فوق.

لنفترض أنك تريد مراقبة القيم الدنيا والقصوى المطلقة لجميع مكونات كل مكالمة 'forward' لدفعة معينة، ولا تفعل ذلك إلا للدفعتين 1 و 3. ثم تقوم بتنفيذ هذه الفئة كما يلي:

```python
debug_overflow = DebugUnderflowOverflow(model, trace_batch_nums=[1, 3])
```

والآن، سيتم تتبع الدفعتين الكاملتين 1 و 3 باستخدام نفس التنسيق الذي يستخدمه كاشف التدفق تحت أو فوق.

يتم فهرسة الدفعات من الصفر.

هذا مفيد إذا كنت تعلم أن البرنامج يبدأ في التصرف بشكل غير طبيعي بعد رقم دفعة معين، بحيث يمكنك الانتقال مباشرة إلى تلك المنطقة. فيما يلي مثال على الإخراج المقتطع لمثل هذا التكوين:

```
*** بدء رقم الدفعة = 1 ***
القيمة الصغرى المطلقة    القيمة القصوى المطلقة    البيانات الوصفية
shared Embedding
1.01e-06 7.92e+02 weight
0.00e+00 2.47e+04 input[0]
5.36e-05 7.92e+02 output
[...]
decoder.dropout Dropout
1.60e-07 2.27e+01 input[0]
0.00e+00 2.52e+01 output
decoder T5Stack
ليس ناتجًا للموتر
lm_head Linear
1.01e-06 7.92e+02 weight
0.00e+00 1.11e+00 input[0]
6.06e-02 8.39e+1 output
T5ForConditionalGeneration
ليس ناتجًا للموتر

*** بدء رقم الدفعة = 3 ***
القيمة الصغرى المطلقة    القيمة القصوى المطلقة    البيانات الوصفية
shared Embedding
1.01e-06 7.92e+02 weight
0.00e+00 2.78e+04 input[0]
5.36e-05 7.92e+02 output
[...]
```

هنا، ستحصل على عدد كبير من الإطارات التي تم تفريغها - بقدر ما كانت هناك مكالمات للأمام في نموذجك، لذلك قد يكون أو لا يكون ما تريده، ولكنه في بعض الأحيان يمكن أن يكون أسهل في الاستخدام لأغراض التصحيح من مصحح الأخطاء العادي. على سبيل المثال، إذا حدثت مشكلة في رقم الدفعة 150. لذا، يمكنك تفريغ آثار الدفعات 149 و 150 ومقارنة الأرقام التي بدأت تختلف.

يمكنك أيضًا تحديد رقم الدفعة الذي سيتم إيقاف التدريب بعده، باستخدام ما يلي:

```python
debug_overflow = DebugUnderflowOverflow(model, trace_batch_nums=[1, 3], abort_after_batch_num=3)
```