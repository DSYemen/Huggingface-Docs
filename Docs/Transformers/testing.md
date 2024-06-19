# اختبار

لنلقِ نظرة على كيفية اختبار نماذج 🤗 Transformers وكيفية كتابة اختبارات جديدة وتحسين الاختبارات الموجودة.

هناك مجموعتا اختبار في المستودع:

1. `tests` -- اختبارات لواجهة برمجة التطبيقات العامة
2. `examples` -- اختبارات للتطبيقات المختلفة التي ليست جزءًا من واجهة برمجة التطبيقات

## كيفية اختبار المحولات

بمجرد إرسال طلب سحب (PR)، يتم اختباره باستخدام 9 وظائف CircleCi. تتم إعادة اختبار كل التزام جديد لهذا الطلب. يتم تحديد هذه الوظائف في [ملف التكوين](https://github.com/huggingface/transformers/tree/main/.circleci/config.yml)، بحيث يمكنك، إذا لزم الأمر، إعادة إنشاء نفس البيئة على جهازك.

لا تقوم وظائف CI هذه بتشغيل الاختبارات `@slow`.

هناك 3 وظائف تشغيلها [إجراءات GitHub](https://github.com/huggingface/transformers/actions):

- [تكامل مركز الشعلة](https://github.com/huggingface/transformers/tree/main/.github/workflows/github-torch-hub.yml): يتحقق مما إذا كان تكامل مركز الشعلة يعمل.
- [ذاتي الاستضافة (دفع)](https://github.com/huggingface/transformers/tree/main/.github/workflows/self-push.yml): تشغيل الاختبارات السريعة على GPU فقط على الالتزامات على `main`. يتم تشغيله فقط إذا تم تحديث الالتزام على `main` للرمز في أحد المجلدات التالية: `src`، `tests`، `.github` (لمنع التشغيل عند إضافة بطاقات نموذج، دفاتر الملاحظات، إلخ.)
- [منفذ ذاتي الاستضافة](https://github.com/huggingface/transformers/tree/main/.github/workflows/self-scheduled.yml): تشغيل الاختبارات العادية والبطيئة على GPU في `tests` و `examples`:

```bash
RUN_SLOW=1 pytest tests/
RUN_SLOW=1 pytest examples/
```

يمكن ملاحظة النتائج [هنا](https://github.com/huggingface/transformers/actions).

## تشغيل الاختبارات

### اختيار الاختبارات التي سيتم تشغيلها

تتطرق هذه الوثيقة إلى العديد من التفاصيل حول كيفية تشغيل الاختبارات. إذا كنت بعد قراءة كل شيء، لا تزال بحاجة إلى مزيد من التفاصيل، فستجدها [هنا](https://docs.pytest.org/en/latest/usage.html).

فيما يلي بعض أكثر الطرق المفيدة لتشغيل الاختبارات.

تشغيل الكل:

```console
pytest
```

أو:

```bash
make test
```

لاحظ أن الأخير محدد على النحو التالي:

```bash
python -m pytest -n auto --dist=loadfile -s -v ./tests/
```

الذي يخبر pytest بالقيام بما يلي:

- قم بتشغيل أكبر عدد ممكن من عمليات اختبار العمليات مثل النوى المركزية (التي قد تكون كثيرة جدًا إذا لم يكن لديك الكثير من ذاكرة الوصول العشوائي!)
- التأكد من أن جميع الاختبارات من نفس الملف سيتم تشغيلها بواسطة نفس عملية الاختبار
- لا تلتقط الإخراج
- تشغيل في الوضع التفصيلي

### الحصول على قائمة بجميع الاختبارات

جميع اختبارات مجموعة الاختبارات:

```bash
pytest --collect-only -q
```

جميع اختبارات ملف اختبار معين:

```bash
pytest tests/test_optimization.py --collect-only -q
```

### تشغيل وحدة اختبار محددة

لتشغيل وحدة اختبار فردية:

```bash
pytest tests/utils/test_logging.py
```

### تشغيل اختبارات محددة

نظرًا لاستخدام unittest داخل معظم الاختبارات، لتشغيل اختبارات فرعية محددة، تحتاج إلى معرفة اسم فئة unittest التي تحتوي على تلك الاختبارات. على سبيل المثال، يمكن أن يكون:

```bash
pytest tests/test_optimization.py::OptimizationTest::test_adam_w
```

هنا:

- `tests/test_optimization.py` - الملف مع الاختبارات
- `OptimizationTest` - اسم الفئة
- `test_adam_w` - اسم دالة الاختبار المحددة

إذا احتوى الملف على عدة فئات، فيمكنك اختيار تشغيل اختبارات فئة معينة فقط. على سبيل المثال:

```bash
pytest tests/test_optimization.py::OptimizationTest
```

سيقوم بتشغيل جميع الاختبارات داخل تلك الفئة.

كما ذكرنا سابقًا، يمكنك معرفة الاختبارات الموجودة داخل فئة `OptimizationTest` عن طريق تشغيل:

```bash
pytest tests/test_optimization.py::OptimizationTest --collect-only -q
```

يمكنك تشغيل الاختبارات بواسطة عبارات مفتاحية.

لتشغيل الاختبارات التي يحتوي اسمها على `adam` فقط:

```bash
pytest -k adam tests/test_optimization.py
```

يمكن استخدام `and` المنطقية و`or` للإشارة إلى ما إذا كان يجب مطابقة جميع الكلمات الرئيسية أو أي منها. يمكن استخدام `not` لنفي.

لتشغيل جميع الاختبارات باستثناء تلك التي يحتوي اسمها على `adam`:

```bash
pytest -k "not adam" tests/test_optimization.py
```

ويمكنك الجمع بين النمطين في واحد:

```bash
pytest -k "ada and not adam" tests/test_optimization.py
```

على سبيل المثال، لتشغيل كل من `test_adafactor` و`test_adam_w`، يمكنك استخدام:

```bash
pytest -k "test_adafactor or test_adam_w" tests/test_optimization.py
```

لاحظ أننا نستخدم `or` هنا، لأننا نريد مطابقة أي من الكلمات الرئيسية لتضمين الاثنين.

إذا كنت تريد تضمين الاختبارات التي تحتوي على كلا النمطين فقط، فيجب استخدام `and`:

```bash
pytest -k "test and ada" tests/test_optimization.py
```

### تشغيل اختبارات `accelerate`

في بعض الأحيان، تحتاج إلى تشغيل اختبارات `accelerate` على نماذجك. للقيام بذلك، يمكنك فقط إضافة `-m accelerate_tests` إلى أمرك، إذا كنت تريد تشغيل هذه الاختبارات على `OPT`، فقم بتشغيل:

```bash
RUN_SLOW=1 pytest -m accelerate_tests tests/models/opt/test_modeling_opt.py
```

### تشغيل اختبارات التوثيق

لاختبار ما إذا كانت أمثلة التوثيق صحيحة، فيجب عليك التحقق من أن `doctests` ناجحة.

كمثال، دعنا نستخدم [docstring لـ `WhisperModel.forward`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/modeling_whisper.py#L1017-L1035):

```python
r"""
Returns:

Example:
```python
>>> import torch
>>> from transformers import WhisperModel, WhisperFeatureExtractor
>>> from datasets import load_dataset
>>> model = WhisperModel.from_pretrained("openai/whisper-base")
>>> feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-base")
>>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
>>> inputs = feature_extractor(ds[0]["audio"]["array"], return_tensors="pt")
>>> input_features = inputs.input_features
>>> decoder_input_ids = torch.tensor([[1, 1]]) * model.config.decoder_start_token_id
>>> last_hidden_state = model(input_features, decoder_input_ids=decoder_input_ids).last_hidden_state
>>> list(last_hidden_state.shape)
[1, 2, 512]
```"""
```

قم ببساطة بتشغيل السطر التالي لاختبار كل مثال docstring في الملف المطلوب تلقائيًا:

```bash
pytest --doctest-modules <path_to_file_or_dir>
```

إذا كان للملف ملحق markdown، فيجب إضافة الحجة `--doctest-glob="*.md"`.

### تشغيل الاختبارات المعدلة فقط

يمكنك تشغيل الاختبارات المتعلقة بالملفات غير المرحلية أو الفرع الحالي (وفقًا لـ Git) باستخدام [pytest-picked](https://github.com/anapaulagomes/pytest-picked). هذه طريقة رائعة لاختبار ما إذا كانت التغييرات التي أجريتها لم تكسر أي شيء بسرعة، حيث لن تقوم بتشغيل الاختبارات المتعلقة بالملفات التي لم تلمسها.

```bash
pip install pytest-picked
```

```bash
pytest --picked
```

سيتم تشغيل جميع الاختبارات من الملفات والمجلدات التي تم تعديلها ولكن لم يتم ارتكابها بعد.

### إعادة تشغيل الاختبارات التي فشلت تلقائيًا عند تعديل المصدر

يوفر [pytest-xdist](https://github.com/pytest-dev/pytest-xdist) ميزة مفيدة جدًا تتمثل في اكتشاف جميع الاختبارات الفاشلة، ثم الانتظار حتى تقوم بتعديل الملفات وإعادة تشغيل تلك الاختبارات الفاشلة باستمرار حتى تمر أثناء إصلاحها. لذلك، لا تحتاج إلى إعادة تشغيل pytest بعد إجراء الإصلاح. يتم تكرار ذلك حتى تمر جميع الاختبارات، وبعد ذلك يتم إجراء تشغيل كامل مرة أخرى.

```bash
pip install pytest-xdist
```

للانتقال إلى الوضع: `pytest -f` أو `pytest --looponfail`

يتم اكتشاف تغييرات الملف عن طريق البحث في دلائل `looponfailroots` وجميع محتوياتها (بشكل متكرر).

إذا لم ينجح الافتراضي لهذا القيمة معك، فيمكنك تغييره في مشروعك عن طريق تعيين خيار تكوين في `setup.cfg`:

```ini
[tool:pytest]
looponfailroots = transformers tests
```

أو ملفات `pytest.ini` / `tox.ini`:

```ini
[pytest]
looponfailroots = transformers tests
```

سيؤدي هذا إلى البحث فقط عن تغييرات الملفات في الدلائل المقابلة، المحددة نسبيًا إلى دليل الملف ini.

[pytest-watch](https://github.com/joeyespo/pytest-watch) هو تنفيذ بديل لهذه الوظيفة.

### تخطي وحدة اختبار

إذا كنت تريد تشغيل جميع وحدات الاختبار، باستثناء عدد قليل منها، فيمكنك استبعادها عن طريق إعطاء قائمة صريحة بالاختبارات التي سيتم تشغيلها. على سبيل المثال، لتشغيل الكل باستثناء اختبارات `test_modeling_*.py`:

```bash
pytest *ls -1 tests/*py | grep -v test_modeling*
```

### مسح الحالة

يجب مسح ذاكرة التخزين المؤقت في عمليات البناء CI وعندما تكون العزلة مهمة (ضد السرعة):

```bash
pytest --cache-clear tests
```

### تشغيل الاختبارات بالتوازي

كما ذكرنا سابقًا، يقوم `make test` بتشغيل الاختبارات بالتوازي عبر المكون الإضافي `pytest-xdist` (`-n X` argument، على سبيل المثال `-n 2` لتشغيل وظيفتين متوازيتين).

يسمح خيار `--dist=` لـ `pytest-xdist` بالتحكم في كيفية تجميع الاختبارات. يضع `--dist=loadfile` الاختبارات الموجودة في ملف واحد في نفس العملية.

نظرًا لأن ترتيب الاختبارات المنفذة مختلف ولا يمكن التنبؤ به، إذا أدى تشغيل مجموعة الاختبارات باستخدام `pytest-xdist` إلى حدوث فشل (مما يعني أن لدينا بعض الاختبارات المقترنة غير المكتشفة)، فاستخدم [pytest-replay](https://github.com/ESSS/pytest-replay) لإعادة تشغيل الاختبارات بنفس الترتيب، والذي يجب أن يساعد بعد ذلك بطريقة ما على تقليل تلك التسلسل الفاشل إلى الحد الأدنى.
### تكرار الاختبار والترتيب

من الجيد تكرار الاختبارات عدة مرات، بشكل متتابع أو عشوائي أو في مجموعات، للكشف عن أي علاقات اعتماد محتملة وعن الأخطاء المتعلقة بالحالة (الإعداد). والتكرار المتعدد والمباشر مفيد أيضاً للكشف عن بعض المشكلات التي تكشفها العشوائية في DL.

#### تكرار الاختبارات

- [pytest-flakefinder](https://github.com/dropbox/pytest-flakefinder):

```bash
pip install pytest-flakefinder
```

ثم قم بتشغيل كل اختبار عدة مرات (50 بشكل افتراضي):

```bash
pytest --flake-finder --flake-runs=5 tests/test_failing_test.py
```

<Tip>
هذه الإضافة لا تعمل مع العلامة -n من pytest-xdist.
</Tip>

<Tip>
هناك إضافة أخرى باسم pytest-repeat، لكنها لا تعمل مع unittest.
</Tip>

#### تشغيل الاختبارات بترتيب عشوائي

```bash
pip install pytest-random-order
```

مهم: وجود pytest-random-order سيقوم تلقائيًا بتعشيق الاختبارات، ولا يلزم إجراء أي تغييرات في التهيئة أو خيارات سطر الأوامر.

كما تم شرحه سابقًا، يسمح ذلك بالكشف عن الاختبارات المقترنة - حيث تؤثر حالة أحد الاختبارات على حالة اختبار آخر. عندما يتم تثبيت pytest-random-order، فإنه يقوم بطباعة البذرة العشوائية التي استخدمها لتلك الجلسة، على سبيل المثال:

```bash
pytest tests
[...]
Using --random-order-bucket=module
Using --random-order-seed=573663
```

حتى أنه إذا فشلت سلسلة معينة، يمكنك إعادة إنتاجها عن طريق إضافة تلك البذرة المحددة، على سبيل المثال:

```bash
pytest --random-order-seed=573663
[...]
Using --random-order-bucket=module
Using --random-order-seed=573663
```

سيقوم بإعادة إنتاج الترتيب المحدد فقط إذا استخدمت قائمة الاختبارات نفسها (أو عدم وجود قائمة على الإطلاق). بمجرد أن تبدأ في تضييق القائمة يدويًا، لا يمكنك الاعتماد على البذرة بعد الآن، ولكن يجب عليك إدراجها يدويًا بالترتيب الدقيق الذي فشلت فيه وإخبار pytest بعدم تعشيقها بدلاً من ذلك باستخدام --random-order-bucket=none، على سبيل المثال:

```bash
pytest --random-order-bucket=none tests/test_a.py tests/test_c.py tests/test_b.py
```

لإيقاف التعشيق لجميع الاختبارات:

```bash
pytest --random-order-bucket=none
```

بشكل افتراضي، يتم ضمنيًا استخدام --random-order-bucket=module، والذي سيقوم بخلط الملفات على مستوى الوحدات النمطية. يمكنه أيضًا الخلط على مستويات "class" و "package" و "global" و "none". للاطلاع على التفاصيل الكاملة، يرجى الاطلاع على [وثائقه](https://github.com/jbasko/pytest-random-order).

بديل آخر للتعشيق هو: [pytest-randomly](https://github.com/pytest-dev/pytest-randomly). تحتوي هذه الوحدة على وظائف/واجهة مماثلة جدًا، ولكنها لا تحتوي على أوضاع الدلاء المتاحة في pytest-random-order. لديها نفس المشكلة في فرض نفسها بمجرد تثبيتها.

### اختلافات المظهر والشعور

#### pytest-sugar

[pytest-sugar](https://github.com/Frozenball/pytest-sugar) هي إضافة تحسن المظهر والشعور، وتضيف شريط تقدم، وتظهر الاختبارات التي تفشل والتأكيد على الفور. يتم تنشيطها تلقائيًا عند التثبيت.

```bash
pip install pytest-sugar
```

لتشغيل الاختبارات بدونها، قم بتشغيل:

```bash
pytest -p no:sugar
```

أو قم بإلغاء تثبيتها.

#### الإبلاغ عن اسم كل اختبار فرعي وتقدمه

لاختبار واحد أو مجموعة من الاختبارات عبر pytest (بعد pip install pytest-pspec):

```bash
pytest --pspec tests/test_optimization.py
```

#### إظهار الاختبارات الفاشلة على الفور

[pytest-instafail](https://github.com/pytest-dev/pytest-instafail) تعرض حالات الفشل والأخطاء على الفور بدلاً من الانتظار حتى نهاية جلسة الاختبار.

```bash
pip install pytest-instafail
```

```bash
pytest --instafail
```

### إلى GPU أو عدم استخدام GPU

في إعداد ممكّن لـ GPU، لاختبار وضع CPU فقط، أضف CUDA_VISIBLE_DEVICES="":

```bash
CUDA_VISIBLE_DEVICES="" pytest tests/utils/test_logging.py
```

أو إذا كان لديك عدة وحدات معالجة رسومية، فيمكنك تحديد أي منها سيتم استخدامها بواسطة pytest. على سبيل المثال، لاستخدام وحدة معالجة الرسومات الثانية إذا كان لديك وحدات معالجة الرسومات 0 و 1، يمكنك تشغيل:

```bash
CUDA_VISIBLE_DEVICES="1" pytest tests/utils/test_logging.py
```

هذا مفيد عندما تريد تشغيل مهام مختلفة على وحدات معالجة رسومية مختلفة.

يجب تشغيل بعض الاختبارات على CPU فقط، والبعض الآخر على CPU أو GPU أو TPU، والبعض الآخر على وحدات معالجة رسومية متعددة. يتم استخدام زخارف التخطي التالية لتحديد متطلبات الاختبارات من حيث CPU/GPU/TPU:

- `require_torch` - سيتم تشغيل هذا الاختبار فقط في ظل وجود وحدة PyTorch
- `require_torch_gpu` - مثل require_torch بالإضافة إلى أنه يتطلب وجود وحدة معالجة رسومية واحدة على الأقل
- `require_torch_multi_gpu` - مثل require_torch بالإضافة إلى أنه يتطلب وجود وحدتي معالجة رسومية على الأقل
- `require_torch_non_multi_gpu` - مثل require_torch بالإضافة إلى أنه يتطلب 0 أو 1 وحدة معالجة رسومية
- `require_torch_up_to_2_gpus` - مثل require_torch بالإضافة إلى أنه يتطلب 0 أو 1 أو 2 وحدة معالجة رسومية
- `require_torch_xla` - مثل require_torch بالإضافة إلى أنه يتطلب وجود وحدة TPU واحدة على الأقل

دعونا نصور متطلبات وحدة معالجة الرسومات في الجدول التالي:

| عدد وحدات معالجة الرسومات | الزخرفة |
| --- | --- |
| >= 0 | `@require_torch` |
| >= 1 | `@require_torch_gpu` |
| >= 2 | `@require_torch_multi_gpu` |
| <2 | `@require_torch_non_multi_gpu` |
| <3 | `@require_torch_up_to_2_gpus` |

على سبيل المثال، هنا اختبار يجب تشغيله فقط عندما تكون هناك وحدتي معالجة رسومية أو أكثر و PyTorch مثبتة:

```python no-style
@require_torch_multi_gpu
def test_example_with_multi_gpu():
```

إذا كان الاختبار يتطلب tensorflow، فاستخدم زخرفة require_tf. على سبيل المثال:

```python no-style
@require_tf
def test_tf_thing_with_tensorflow():
```

يمكن تكديس هذه الزخارف. على سبيل المثال، إذا كان الاختبار بطيئًا ويتطلب وجود وحدة معالجة رسومية واحدة على الأقل في PyTorch، فهذه هي طريقة إعداده:

```python no-style
@require_torch_gpu
@slow
def test_example_slow_on_gpu():
```

بعض الزخارف مثل `@parametrized` تعيد كتابة أسماء الاختبارات، لذلك يجب إدراج زخارف التخطي @require_* في النهاية لكي تعمل بشكل صحيح. فيما يلي مثال على الاستخدام الصحيح:

```python no-style
@parameterized.expand(...)
@require_torch_multi_gpu
def test_integration_foo():
```

لا توجد مشكلة الترتيب هذه مع @pytest.mark.parametrize، يمكنك وضعها أولاً أو آخرًا وستظل تعمل. ولكنه يعمل فقط مع non-unittests.

داخل الاختبارات:

- عدد وحدات معالجة الرسومات المتاحة:

```python
from transformers.testing_utils import get_gpu_count

n_gpu = get_gpu_count() # تعمل مع PyTorch و TensorFlow
```

### الاختبار باستخدام PyTorch خلفية أو جهاز محدد

لتشغيل مجموعة الاختبارات على جهاز PyTorch محدد، أضف TRANSFORMERS_TEST_DEVICE="$device" حيث $device هو الخلفية المستهدفة. على سبيل المثال، لاختبار وضع CPU فقط:

```bash
TRANSFORMERS_TEST_DEVICE="cpu" pytest tests/utils/test_logging.py
```

تكون هذه المتغيرات مفيدة لاختبار خلفيات PyTorch مخصصة أو أقل شيوعًا مثل mps أو xpu أو npu. يمكن استخدامه أيضًا لتحقيق نفس تأثير CUDA_VISIBLE_DEVICES عن طريق استهداف وحدات معالجة رسومية محددة أو الاختبار في وضع CPU فقط.

قد تتطلب بعض الأجهزة استيرادًا إضافيًا بعد استيراد PyTorch للمرة الأولى. يمكن تحديد ذلك باستخدام متغير البيئة TRANSFORMERS_TEST_BACKEND:

```bash
TRANSFORMERS_TEST_BACKEND="torch_npu" pytest tests/utils/test_logging.py
```

قد تتطلب الخلفيات البديلة أيضًا استبدال وظائف محددة للجهاز. على سبيل المثال، قد تحتاج torch.cuda.manual_seed إلى استبدالها بوظيفة تعيين بذور محددة للجهاز مثل torch.npu.manual_seed أو torch.xpu.manual_seed لتعيين بذرة عشوائية بشكل صحيح على الجهاز. لتحديد خلفية جديدة مع وظائف محددة للجهاز عند تشغيل مجموعة الاختبارات، قم بإنشاء ملف مواصفات Python باسم spec.py بالتنسيق التالي:

```python
import torch
import torch_npu # بالنسبة إلى xpu، استبدلها بـ `import intel_extension_for_pytorch`
# !! يمكن إضافة استيرادات إضافية هنا !!

# تحديد اسم الجهاز (على سبيل المثال 'cuda' أو 'cpu' أو 'npu' أو 'xpu' أو 'mps')
DEVICE_NAME = 'npu'

# تحديد الخلفيات المحددة للجهاز للتبديل إليها.
# إذا لم يتم تحديدها، فسيتم الرجوع إلى 'default' في 'testing_utils.py`
MANUAL_SEED_FN = torch.npu.manual_seed
EMPTY_CACHE_FN = torch.npu.empty_cache
DEVICE_COUNT_FN = torch.npu.device_count
```

يسمح هذا التنسيق أيضًا بتحديد أي استيرادات إضافية مطلوبة. لاستخدام هذا الملف لاستبدال الطرق المكافئة في مجموعة الاختبارات، قم بتعيين متغير البيئة TRANSFORMERS_TEST_DEVICE_SPEC إلى مسار ملف المواصفات، على سبيل المثال TRANSFORMERS_TEST_DEVICE_SPEC=spec.py.

حاليًا، يتم دعم MANUAL_SEED_FN و EMPTY_CACHE_FN و DEVICE_COUNT_FN فقط للتبديل المحدد للجهاز.

### التدريب الموزع

لا يمكن لـ pytest التعامل مع التدريب الموزع مباشرةً. إذا تم محاولة ذلك - فإن العمليات الفرعية لا تقوم بالشيء الصحيح وتنتهي بالتفكير في أنها pytest وتبدأ في تشغيل مجموعة الاختبارات في حلقات. ولكنه يعمل إذا قام أحد بتشغيل عملية عادية تقوم بعد ذلك بتشغيل العديد من العمال وإدارة أنابيب الإدخال/الإخراج.

فيما يلي بعض الاختبارات التي تستخدمها:

- [test_trainer_distributed.py](https://github.com/huggingface/transformers/tree/main/tests/trainer/test_trainer_distributed.py)
- [test_deepspeed.py](https://github.com/huggingface/transformers/tree/main/tests/deepspeed/test_deepspeed.py)

للانتقال مباشرةً إلى نقطة التنفيذ، ابحث عن مكالمة execute_subprocess_async في تلك الاختبارات.

ستحتاج إلى وحدتي معالجة رسومية على الأقل لرؤية هذه الاختبارات قيد التنفيذ:

```bash
CUDA_VISIBLE_DEVICES=0,1 RUN_SLOW=1 pytest -sv tests/test_trainer_distributed.py
```

### التقاط الإخراج

أثناء تنفيذ الاختبار، يتم التقاط أي إخراج يتم إرساله إلى stdout و stderr. إذا فشل أحد الاختبارات أو طريقة الإعداد، فسيتم عادةً عرض الإخراج المقابل الذي تم التقاطه جنبًا إلى جنب مع تعقب الفشل.

لإيقاف التقاط الإخراج والحصول على stdout و stderr بشكل طبيعي، استخدم -s أو --capture=no:

```bash
pytest -s tests/utils/test_logging.py
```

لإرسال نتائج الاختبار إلى إخراج بتنسيق JUnit:

```bash
pytest tests --junitxml=result.xml
```

### التحكم في الألوان

لعدم استخدام الألوان (على سبيل المثال، الأصفر على خلفية بيضاء غير قابل للقراءة):

```bash
pytest --color=no tests/utils/test_logging.py
```
### إرسال تقرير الاختبار إلى خدمة Pastebin عبر الإنترنت

إنشاء عنوان URL لكل فشل في الاختبار:

```bash
pytest --pastebin=failed tests/utils/test_logging.py
```

سيقوم هذا الأمر بإرسال معلومات تشغيل الاختبار إلى خدمة Paste عن بُعد وتوفير عنوان URL لكل فشل. يمكنك اختيار الاختبارات كالمعتاد أو إضافة -x على سبيل المثال إذا كنت تريد إرسال فشل معين فقط.

إنشاء عنوان URL لسجل جلسة الاختبار بالكامل:

```bash
pytest --pastebin=all tests/utils/test_logging.py
```

## كتابة الاختبارات

تعتمد اختبارات 🤗 transformers على `unittest`، ولكن يتم تشغيلها بواسطة `pytest`، لذلك يمكن استخدام ميزات كلا النظامين في معظم الوقت.

يمكنك قراءة [هنا](https://docs.pytest.org/en/stable/unittest.html) لمعرفة الميزات المدعومة، ولكن الشيء المهم الذي يجب تذكره هو أن معظم مؤشرات `pytest` لا تعمل. ولا التصنيف أيضًا، ولكننا نستخدم الوحدة النمطية `parameterized` التي تعمل بطريقة مشابهة.

### التصنيف

غالبًا ما تكون هناك حاجة لتشغيل نفس الاختبار عدة مرات، ولكن بحجج مختلفة. يمكن القيام بذلك من داخل الاختبار، ولكن لا توجد طريقة لتشغيل هذا الاختبار لمجموعة واحدة من الحجج فقط.

```python
# test_this1.py
import unittest
from parameterized import parameterized


class TestMathUnitTest(unittest.TestCase):
    @parameterized.expand(
    [
    ("negative", -1.5, -2.0),
    ("integer", 1, 1.0),
    ("large fraction", 1.6, 1),
    ]
    )
    def test_floor(self, name, input, expected):
    assert_equal(math.floor(input), expected)
```

الآن، بشكل افتراضي، سيتم تشغيل هذا الاختبار 3 مرات، وفي كل مرة يتم تعيين الحجج الثلاث الأخيرة لـ `test_floor` إلى الحجج المقابلة في قائمة المعلمات.

ويمكنك تشغيل مجموعات "negative" و "integer" من المعلمات فقط باستخدام:

```bash
pytest -k "negative and integer" tests/test_mytest.py
```

أو جميع المجموعات الفرعية باستثناء "negative" باستخدام:

```bash
pytest -k "not negative" tests/test_mytest.py
```

بالإضافة إلى استخدام عامل تصفية `-k` الذي تم ذكره للتو، يمكنك معرفة الاسم الدقيق لكل اختبار فرعي وتشغيل أي منها أو جميعها باستخدام أسمائها الدقيقة.

```bash
pytest test_this1.py --collect-only -q
```

وسوف يسرد:

```bash
test_this1.py::TestMathUnitTest::test_floor_0_negative
test_this1.py::TestMathUnitTest::test_floor_1_integer
test_this1.py::TestMathUnitTest::test_floor_2_large_fraction
```

لذلك الآن يمكنك تشغيل اختبارين فرعيين محددين فقط:

```bash
pytest test_this1.py::TestMathUnitTest::test_floor_0_negative test_this1.py::TestMathUnitTest::test_floor_1_integer
```

تعمل الوحدة النمطية [parameterized](https://pypi.org/project/parameterized/)، والتي توجد بالفعل في التبعيات التنموية لـ `transformers`، لكل من: `unittests` و `pytest` tests.

ومع ذلك، إذا لم يكن الاختبار عبارة عن `unittest`، فيمكنك استخدام `pytest.mark.parametrize` (أو قد ترى أنها مستخدمة في بعض الاختبارات الموجودة، خاصةً تحت `examples`).

فيما يلي نفس المثال، ولكن هذه المرة باستخدام مؤشر `parametrize` من `pytest`:

```python
# test_this2.py
import pytest


@pytest.mark.parametrize(
"name, input, expected",
[
("negative", -1.5, -2.0),
("integer", 1, 1.0),
("large fraction", 1.6, 1),
],
)
def test_floor(name, input, expected):
assert_equal(math.floor(input), expected)
```

مثل `parameterized`، يمكنك التحكم الدقيق في الاختبارات الفرعية التي يتم تشغيلها باستخدام عامل تصفية `-k`، إلا أن دالة التصنيف هذه تُنشئ مجموعة مختلفة قليلاً من الأسماء للاختبارات الفرعية. فيما يلي كيفية ظهورها:

```bash
pytest test_this2.py --collect-only -q
```

وسوف يسرد:

```bash
test_this2.py::test_floor[integer-1-1.0]
test_this2.py::test_floor[negative--1.5--2.0]
test_this2.py::test_floor[large fraction-1.6-1]
```

لذلك الآن يمكنك تشغيل الاختبار المحدد فقط:

```bash
pytest test_this2.py::test_floor[negative--1.5--2.0] test_this2.py::test_floor[integer-1-1.0]
```

كما في المثال السابق.

### الملفات والمجلدات

في الاختبارات، نحتاج غالبًا إلى معرفة مكان وجود الأشياء بالنسبة لملف الاختبار الحالي، وهذا ليس أمرًا بسيطًا لأن الاختبار قد يتم استدعاؤه من أكثر من دليل واحد أو قد يكون موجودًا في مجلدات فرعية بدرجات مختلفة. تقوم فئة المساعدة `transformers.test_utils.TestCasePlus` بحل هذه المشكلة عن طريق فرز جميع المسارات الأساسية وتوفير وصول سهل إليها:

- كائنات `pathlib` (جميعها محددة بالكامل):
   - `test_file_path` - مسار ملف الاختبار الحالي، أي `__file__`
   - `test_file_dir` - الدليل الذي يحتوي على ملف الاختبار الحالي
   - `tests_dir` - دليل مجموعة الاختبارات `tests`
   - `examples_dir` - دليل مجموعة الاختبارات `examples`
   - `repo_root_dir` - دليل مستودع
   - `src_dir` - دليل `src` (أي حيث يوجد المجلد الفرعي `transformers`)

- المسارات المعبرة عن السلاسل النصية---نفس ما سبق ولكن هذه الأساليب تعيد المسارات كسلاسل نصية، بدلاً من كائنات `pathlib`:
   - `test_file_path_str`
   - `test_file_dir_str`
   - `tests_dir_str`
   - `examples_dir_str`
   - `repo_root_dir_str`
   - `src_dir_str`

للبدء في استخدام تلك الطرق، كل ما تحتاجه هو التأكد من أن الاختبار موجود في فئة فرعية من `transformers.test_utils.TestCasePlus`. على سبيل المثال:

```python
from transformers.testing_utils import TestCasePlus


class PathExampleTest(TestCasePlus):
    def test_something_involving_local_locations(self):
    data_dir = self.tests_dir / "fixtures/tests_samples/wmt_en_ro"
```

إذا لم تكن بحاجة إلى التعامل مع المسارات عبر `pathlib` أو إذا كنت بحاجة فقط إلى مسار كسلسلة نصية، فيمكنك دائمًا استدعاء `str()` على كائن `pathlib` أو استخدام الطرق التي تنتهي بـ `_str`. على سبيل المثال:

```python
from transformers.testing_utils import TestCasePlus


class PathExampleTest(TestCasePlus):
    def test_something_involving_stringified_locations(self):
    examples_dir = self.examples_dir_str
```

### الملفات والمجلدات المؤقتة

يعد استخدام الملفات والمجلدات المؤقتة الفريدة أمرًا ضروريًا لتشغيل الاختبارات بشكل متوازي، بحيث لا تكتب الاختبارات فوق بيانات بعضها البعض. كما نريد حذف الملفات والمجلدات المؤقتة في نهاية كل اختبار قام بإنشائها. لذلك، من الضروري استخدام حزم مثل `tempfile`، والتي تلبي هذه الاحتياجات.

ومع ذلك، عند تصحيح أخطاء الاختبارات، تحتاج إلى القدرة على رؤية ما يتم إدخاله في الملف المؤقت أو المجلد وتريد معرفة مساره الدقيق وعدم جعله عشوائيًا في كل مرة يتم فيها إعادة تشغيل الاختبار.

تعد فئة المساعدة `transformers.test_utils.TestCasePlus` أفضل للاستخدام في مثل هذه الأغراض. إنها فئة فرعية من `unittest.TestCase`، لذلك يمكننا بسهولة أن نرث منها في وحدات الاختبار.

فيما يلي مثال على استخدامها:

```python
from transformers.testing_utils import TestCasePlus


class ExamplesTests(TestCasePlus):
    def test_whatever(self):
    tmp_dir = self.get_auto_remove_tmp_dir()
```

ينشئ هذا الكود مجلدًا مؤقتًا فريدًا، ويحدد `tmp_dir` إلى موقعه.

- إنشاء مجلد مؤقت فريد:

```python
def test_whatever(self):
tmp_dir = self.get_auto_remove_tmp_dir()
```

سيحتوي `tmp_dir` على مسار المجلد المؤقت الذي تم إنشاؤه. وسيتم إزالته تلقائيًا في نهاية الاختبار.

- إنشاء مجلد مؤقت من اختياري، والتأكد من أنه فارغ قبل بدء الاختبار وعدم إفراغه بعد الاختبار.

```python
def test_whatever(self):
tmp_dir = self.get_auto_remove_tmp_dir("./xxx")
```

هذا مفيد للتصحيح عندما تريد مراقبة مجلد محدد والتأكد من أن الاختبارات السابقة لم تترك أي بيانات فيه.

- يمكنك تجاوز السلوك الافتراضي عن طريق تجاوز وسيطي `before` و `after` مباشرةً، مما يؤدي إلى أحد السلوكيات التالية:
   - `before=True`: سيتم دائمًا مسح المجلد المؤقت في بداية الاختبار.
   - `before=False`: إذا كان المجلد المؤقت موجودًا بالفعل، فستظل أي ملفات موجودة فيه.
   - `after=True`: سيتم دائمًا حذف المجلد المؤقت في نهاية الاختبار.
   - `after=False`: سيتم دائمًا ترك المجلد المؤقت دون تغيير في نهاية الاختبار.

<Tip>

لتشغيل ما يعادل `rm -r` بأمان، يُسمح فقط بالمجلدات الفرعية لشجرة التحقق من مستودع المشروع إذا تم استخدام مجلد `tmp_dir` صريح، بحيث لا يتم عن طريق الخطأ مسح أي جزء مهم من نظام الملفات مثل `/tmp` أو ما شابه ذلك. يرجى دائمًا تمرير المسارات التي تبدأ بـ `./`.

</Tip>

<Tip>

يمكن لكل اختبار تسجيل عدة مجلدات مؤقتة وسيتم إزالتها تلقائيًا، ما لم يُطلب خلاف ذلك.

</Tip>

### تجاوز مؤقت لـ sys.path

إذا كنت بحاجة إلى تجاوز مؤقت لـ `sys.path` لاستيراد اختبار آخر، على سبيل المثال، فيمكنك استخدام إدارة السياق `ExtendSysPath`. مثال:

```python
import os
from transformers.testing_utils import ExtendSysPath

bindir = os.path.abspath(os.path.dirname(__file__))
with ExtendSysPath(f"{bindir}/.."):
from test_trainer import TrainerIntegrationCommon  # noqa
```
### تجاوز الاختبارات

هذا مفيد عندما يتم اكتشاف خطأ ويتم كتابة اختبار جديد، ولكن لم يتم إصلاح الخطأ بعد. من أجل أن نتمكن من الالتزام به في المستودع الرئيسي، نحتاج إلى التأكد من أنه يتم تخطيه أثناء "إجراء الاختبار".

الطرق:

- يعني **التجاوز** أنك تتوقع أن يمر اختبارك فقط إذا تم استيفاء بعض الشروط، وإلا فيجب على pytest تخطي تشغيل الاختبار بالكامل. ومن الأمثلة الشائعة على ذلك تخطي الاختبارات الخاصة بنظام Windows فقط على منصات غير Windows، أو تخطي الاختبارات التي تعتمد على مورد خارجي غير متوفر في الوقت الحالي (مثل قاعدة بيانات).

- يعني **xfail** أنك تتوقع فشل الاختبار لسبب ما. ومن الأمثلة الشائعة على ذلك اختبار ميزة لم يتم تنفيذها بعد، أو خطأ لم يتم إصلاحه بعد. عندما ينجح الاختبار على الرغم من توقع فشله (معلمة pytest.mark.xfail)، فهو xpass وسيتم الإبلاغ عنه في ملخص الاختبار.

أحد الاختلافات المهمة بين الاثنين هو أن "التجاوز" لا يشغل الاختبار، و"xfail" يفعل ذلك. لذا إذا كان الكود الذي يحتوي على أخطاء يسبب حالة سيئة ستؤثر على الاختبارات الأخرى، فلا تستخدم "xfail".

#### التنفيذ

- إليك كيفية تخطي الاختبار بالكامل دون قيد أو شرط:

```python
@unittest.skip("هذه المشكلة بحاجة إلى إصلاح")
def test_feature_x():
```

أو عبر pytest:

```python
@pytest.mark.skip(reason="هذه المشكلة بحاجة إلى إصلاح")
```

أو بطريقة "xfail":

```python
@pytest.mark.xfail
def test_feature_x():
```

فيما يلي كيفية تخطي اختبار بناءً على فحوصات داخلية داخل الاختبار:

```python
def test_feature_x():
    if not has_something():
        pytest.skip("تكوين غير مدعوم")
```

أو الوحدة النمطية بأكملها:

```python
import pytest

if not pytest.config.getoption("--custom-flag"):
    pytest.skip("--custom-flag مفقود، يتم تخطي الاختبارات"، allow_module_level=True)
```

أو بطريقة "xfail":

```python
def test_feature_x():
    pytest.xfail("من المتوقع أن يفشل حتى يتم إصلاح الخطأ XYZ")
```

- فيما يلي كيفية تخطي جميع الاختبارات في وحدة نمطية إذا كان هناك استيراد مفقود:

```python
docutils = pytest.importorskip("docutils"، minversion="0.3")
```

- تخطي اختبار بناءً على شرط:

```python
@pytest.mark.skipif(sys.version_info < (3,6)، reason="يتطلب الإصدار 3.6 من Python أو أعلى")
def test_feature_x():
```

أو:

```python
@unittest.skipIf(torch_device == "cpu"، "لا يمكن إجراء الدقة النصفية")
def test_feature_x():
```

أو تخطي الوحدة النمطية بأكملها:

```python
@pytest.mark.skipif(sys.platform == 'win32'، reason="لا يعمل على نظام Windows")
class TestClass():
    def test_feature_x(self):
```

لمزيد من التفاصيل والأمثلة والطرق، انظر [هنا](https://docs.pytest.org/en/latest/skipping.html).

### الاختبارات البطيئة

مكتبة الاختبارات تتزايد باستمرار، ويستغرق بعض الاختبارات دقائق للتشغيل، لذلك لا يمكننا تحمل الانتظار لمدة ساعة حتى تكتمل مجموعة الاختبارات على CI. لذلك، مع بعض الاستثناءات للاختبارات الأساسية، يجب وضع علامة على الاختبارات البطيئة كما هو موضح في المثال أدناه:

```python
from transformers.testing_utils import slow
@slow
def test_integration_foo():
```

بمجرد وضع علامة على الاختبار على أنه `@slow`، لتشغيل هذه الاختبارات، قم بتعيين متغير البيئة `RUN_SLOW=1`، على سبيل المثال:

```bash
RUN_SLOW=1 pytest tests
```

يقوم بعض الديكورات مثل `@parameterized` بإعادة كتابة أسماء الاختبارات، لذلك يجب إدراج `@slow` وبقية الديكورات `@require_*` في النهاية حتى تعمل بشكل صحيح. فيما يلي مثال على الاستخدام الصحيح:

```python
@parameterized.expand(...)
@slow
def test_integration_foo():
```

كما هو موضح في بداية هذه الوثيقة، يتم تشغيل الاختبارات البطيئة وفقًا لجدول زمني، بدلاً من فحوصات CI في PRs. لذا فمن الممكن أن يتم تفويت بعض المشكلات أثناء تقديم PR ودمجها. سيتم اكتشاف هذه المشكلات أثناء مهمة CI المجدولة التالية. ولكن هذا يعني أيضًا أنه من المهم تشغيل الاختبارات البطيئة على جهازك قبل تقديم PR.

فيما يلي آلية صنع القرار لاختيار الاختبارات التي يجب وضع علامة عليها على أنها بطيئة:

إذا كان الاختبار يركز على أحد المكونات الداخلية للمكتبة (مثل ملفات النمذجة أو ملفات التمييز)، فيجب علينا تشغيل هذا الاختبار في مجموعة الاختبارات غير البطيئة. إذا كان يركز على جانب آخر من جوانب المكتبة، مثل الوثائق أو الأمثلة، فيجب علينا تشغيل هذه الاختبارات في مجموعة الاختبارات البطيئة. ثم، لتنقيح هذا النهج، يجب أن تكون لدينا استثناءات:

- يجب وضع علامة على جميع الاختبارات التي تحتاج إلى تنزيل مجموعة كبيرة من الأوزان أو مجموعة بيانات أكبر من 50 ميجابايت (مثل اختبارات تكامل النماذج أو برامج التمييز أو خطوط الأنابيب) على أنها بطيئة. إذا كنت تضيف نموذجًا جديدًا، فيجب عليك إنشاء وتحميل إصدار صغير منه (بأوزان عشوائية) لاختبارات الدمج. يتم مناقشة ذلك في الفقرات التالية.

- يجب وضع علامة على جميع الاختبارات التي تحتاج إلى إجراء تدريب غير مُحسّن خصيصًا ليكون سريعًا على أنها بطيئة.

- يمكننا تقديم استثناءات إذا كانت بعض هذه الاختبارات التي يجب ألا تكون بطيئة بطيئة للغاية، ووضع علامة عليها على أنها `@slow`. تعد اختبارات النمذجة التلقائية، التي تقوم بحفظ وتحميل ملفات كبيرة على القرص، مثالًا جيدًا على الاختبارات التي تم وضع علامة عليها على أنها `@slow`.

- إذا اكتمل الاختبار في أقل من ثانية واحدة على CI (بما في ذلك عمليات التنزيل إن وجدت)، فيجب أن يكون اختبارًا عاديًا بغض النظر عن ذلك.

بشكل جماعي، يجب أن تغطي جميع الاختبارات غير البطيئة المكونات الداخلية المختلفة، مع الحفاظ على سرعتها. على سبيل المثال، يمكن تحقيق تغطية كبيرة من خلال الاختبار باستخدام نماذج صغيرة تم إنشاؤها خصيصًا بأوزان عشوائية. تحتوي هذه النماذج على الحد الأدنى من عدد الطبقات (على سبيل المثال، 2)، وحجم المفردات (على سبيل المثال، 1000)، وما إلى ذلك. بعد ذلك، يمكن لاختبارات `@slow` استخدام نماذج كبيرة وبطيئة لإجراء الاختبارات النوعية. لمشاهدة استخدام هذه، ما عليك سوى البحث عن النماذج "tiny" باستخدام:

```bash
grep tiny tests examples
```

فيما يلي مثال على [سكريبت](https://github.com/huggingface/transformers/tree/main/scripts/fsmt/fsmt-make-tiny-model.py) الذي أنشأ نموذج "tiny" [stas/tiny-wmt19-en-de](https://huggingface.co/stas/tiny-wmt19-en-de). يمكنك ضبطه بسهولة على الهندسة المعمارية المحددة لنموذجك.

من السهل قياس وقت التشغيل بشكل غير صحيح إذا كان هناك، على سبيل المثال، إشراف على تنزيل نموذج ضخم، ولكن إذا قمت باختباره محليًا، فسيتم تخزين الملفات التي تم تنزيلها مؤقتًا، وبالتالي لن يتم قياس وقت التنزيل. لذا تحقق من تقرير سرعة التنفيذ في سجلات CI بدلاً من ذلك (إخراج `pytest --durations=0 tests`).

هذا التقرير مفيد أيضًا للعثور على القيم الشاذة البطيئة التي لم يتم وضع علامة عليها على هذا النحو، أو التي تحتاج إلى إعادة كتابتها لتكون سريعة. إذا لاحظت أن مجموعة الاختبارات بدأت تصبح بطيئة على CI، فسيظهر أعلى قائمة هذا التقرير أبطأ الاختبارات.

### اختبار إخراج stdout/stderr

لاختبار الوظائف التي تكتب في `stdout` و/أو `stderr`، يمكن للاختبار الوصول إلى هذه التدفقات باستخدام نظام [capsys](https://docs.pytest.org/en/latest/capture.html) في pytest. فيما يلي كيفية القيام بذلك:

```python
import sys


def print_to_stdout(s):
    print(s)


def print_to_stderr(s):
    sys.stderr.write(s)


def test_result_and_stdout(capsys):
    msg = "Hello"
    print_to_stdout(msg)
    print_to_stderr(msg)
    out, err = capsys.readouterr() # استهلاك تيارات الإخراج التي تم التقاطها
    # اختياري: إذا كنت تريد إعادة تشغيل التدفقات التي تم استهلاكها:
    sys.stdout.write(out)
    sys.stderr.write(err)
    # الاختبار:
    تأكد من وجود msg في out
    تأكد من وجود msg في err
```

وبالطبع، في معظم الوقت، ستأتي "stderr" كجزء من استثناء، لذا يجب استخدام try/except في مثل هذه الحالة:

```python
def raise_exception(msg):
    raise ValueError(msg)


def test_something_exception():
    msg = "ليست قيمة جيدة"
    error = ""
    try:
        raise_exception(msg)
    except Exception as e:
        error = str(e)
    تأكد من وجود msg في الخطأ، f "{msg} موجود في الاستثناء: \ n {error}"
```

نهج آخر لالتقاط stdout هو عبر `contextlib.redirect_stdout`:

```python
from io import StringIO
from contextlib import redirect_stdout


def print_to_stdout(s):
    print(s)


def test_result_and_stdout():
    msg = "Hello"
    buffer = StringIO()
    with redirect_stdout(buffer):
        print_to_stdout(msg)
    out = buffer.getvalue()
    # اختياري: إذا كنت تريد إعادة تشغيل التدفقات التي تم استهلاكها:
    sys.stdout.write(out)
    # الاختبار:
    تأكد من وجود msg في out
```

مشكلة محتملة مهمة مع التقاط stdout هي أنه قد يحتوي على أحرف `\r` التي تقوم في الطباعة العادية بإعادة تعيين كل ما تم طباعته حتى الآن. لا توجد مشكلة مع pytest، ولكن مع pytest -s يتم تضمين هذه الأحرف في المؤشر، لذا للتمكن من تشغيل الاختبار مع أو بدون -s، يجب عليك إجراء تنظيف إضافي للإخراج الذي تم التقاطه، باستخدام `re.sub(r'~.*\r'، ''، buf، 0، re.M)`.

ولكن بعد ذلك، لدينا فئة سياق مساعد للتعامل مع كل ذلك تلقائيًا، بغض النظر عما إذا كان يحتوي على بعض الأحرف '\r' أم لا، لذا فهو بسيط:

```python
from transformers.testing_utils import CaptureStdout

with CaptureStdout() as cs:
    function_that_writes_to_stdout()
print(cs.out)
```

فيما يلي مثال كامل على الاختبار:

```python
from transformers.testing_utils import CaptureStdout

msg = "رسالة سرية\r"
final = "مرحبا بالعالم"
with CaptureStdout() as cs:
    print(msg + final)
تأكد من أن cs.out == final + "\n"، f "تم التقاطه: {cs.out}، متوقعًا {final}"
```

إذا كنت تريد التقاط "stderr"، فاستخدم فئة "CaptureStderr" بدلاً من ذلك:

```python
from transformers.testing_utils import CaptureStderr

with CaptureStderr() as cs:
    function_that_writes_to_stderr()
print(cs.err)
```

إذا كنت بحاجة إلى التقاط كلا التدفقين في نفس الوقت، فاستخدم فئة "CaptureStd" الأصلية:

```python
from transformers.testing_utils import CaptureStd

with CaptureStd() as cs:
    function_that_writes_to_stdout_and_stderr()
print(cs.err, cs.out)
```

أيضًا، للمساعدة في تصحيح مشكلات الاختبار، تقوم فئات سياق المساعد هذه بشكل افتراضي بإعادة تشغيل التدفقات التي تم التقاطها عند الخروج من السياق.

### التقاط دفق المسجل

إذا كنت بحاجة إلى التحقق من إخراج مسجل، فيمكنك استخدام `CaptureLogger`:

```python
from transformers import logging
from transformers.testing_utils import CaptureLogger

msg = "اختبار 1، 2، 3"
logging.set_verbosity_info()
logger = logging.get_logger("transformers.models.bart.tokenization_bart")
with CaptureLogger(logger) as cl:
    logger.info(msg)
تأكد من أن cl.out، msg + "\n"
```
### إجراء الاختبارات باستخدام متغيرات البيئة

إذا أردت اختبار تأثير متغيرات البيئة لاختبار محدد، فيمكنك استخدام الديكور المساعد `transformers.testing_utils.mockenv`.

```python
from transformers.testing_utils import mockenv

class HfArgumentParserTest(unittest.TestCase):
    @mockenv(TRANSFORMERS_VERBOSITY="error")
    def test_env_override(self):
        env_level_str = os.getenv("TRANSFORMERS_VERBOSITY", None)
```

في بعض الأحيان، يلزم استدعاء برنامج خارجي، مما يتطلب تعيين `PYTHONPATH` في `os.environ` لتشمل عدة مسارات محلية. وتأتي فئة مساعدة `transformers.test_utils.TestCasePlus` للمساعدة:

```python
from transformers.testing_utils import TestCasePlus

class EnvExampleTest(TestCasePlus):
    def test_external_prog(self):
        env = self.get_env()
        # الآن قم باستدعاء البرنامج الخارجي، مع تمرير "env" إليه
```

اعتمادًا على ما إذا كان ملف الاختبار موجودًا في مجموعة اختبارات "tests" أو "examples"، فسيتم تعيين "env[PYTHONPATH]" بشكل صحيح لتشمل أحد هذين الدليلين، وكذلك دليل "src" لضمان إجراء الاختبار مقابل مستودع "repo" الحالي، وأخيرًا مع أي "env[PYTHONPATH]" تم تعيينه بالفعل قبل استدعاء الاختبار إذا كان هناك أي شيء.

تُنشئ طريقة المساعدة هذه نسخة من كائن "os.environ"، بحيث يظل الكائن الأصلي سليمًا.

### الحصول على نتائج قابلة للتكرار

في بعض الحالات، قد ترغب في إزالة العشوائية من اختباراتك. للحصول على نتائج متطابقة وقابلة للتكرار، ستحتاج إلى تثبيت البذرة:

```python
seed = 42

# مولد الأرقام العشوائية في بايثون
import random

random.seed(seed)

# مولدات الأرقام العشوائية في باي تورش
import torch

torch.seed(seed)
torch.backends.cudnn.deterministic = True
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# مولد الأرقام العشوائية في نومبي
import numpy as np

np.random.seed(seed)

# مولد الأرقام العشوائية في تي إف
tf.random.set_seed(seed)
```

### تصحيح الاختبارات

لبدء مصحح الأخطاء عند نقطة التحذير، قم بما يلي:

```bash
pytest tests/utils/test_logging.py -W error::UserWarning --pdb
```

## العمل مع سير عمل إجراءات جيت هاب

لتشغيل مهمة CI ذاتية الدفع، يجب عليك:

1. إنشاء فرع جديد في أصل "transformers" (ليس فرعًا!).
2. يجب أن يبدأ اسم الفرع بـ "ci_" أو "ci-". (يتم تشغيل "main" أيضًا، ولكن لا يمكننا إجراء طلبات سحب "PRs" على "main"). كما يتم تشغيله فقط لمسارات محددة - يمكنك العثور على التعريف المحدث في حالة تغييره منذ كتابة هذه الوثيقة [هنا](https://github.com/huggingface/transformers/blob/main/.github/workflows/self-push.yml) تحت *push:*
3. إنشاء طلب سحب "PR" من هذا الفرع.
4. بعد ذلك، يمكنك رؤية المهمة تظهر [هنا](https://github.com/huggingface/transformers/actions/workflows/self-push.yml). قد لا يتم تشغيله على الفور إذا كان هناك تراكم.

## اختبار ميزات CI التجريبية

يمكن أن يكون اختبار ميزات CI مشكلة محتملة لأنه يمكن أن يتعارض مع وظيفة CI العادية. لذلك، إذا كان سيتم إضافة ميزة CI جديدة، فيجب القيام بذلك على النحو التالي.

1. قم بإنشاء مهمة مخصصة جديدة لاختبار ما يحتاج إلى اختبار.
2. يجب أن تنجح المهمة الجديدة دائمًا حتى تعطينا علامة ✓ خضراء (التفاصيل أدناه).
3. دعها تعمل لبضعة أيام لترى أن مجموعة متنوعة من أنواع طلبات السحب "PRs" المختلفة يتم تشغيلها عليها (فروع مستودعات المستخدمين، والفروع غير المتفرعة، والفروع الناتجة عن تحرير الملف المباشر في واجهة مستخدم "github.com"، والدفعات القسرية المختلفة، إلخ. - هناك الكثير) مع مراقبة سجلات المهمة التجريبية (ليس الوظيفة الكلية الخضراء لأنها خضراء عن قصد دائمًا)
4. عندما يكون كل شيء واضحًا، قم بدمج التغييرات الجديدة في المهام الموجودة.

بهذه الطريقة، لن تتعارض التجارب على وظيفة CI نفسها مع سير العمل العادي.

والآن، كيف يمكننا جعل المهمة تنجح دائمًا بينما يتم تطوير ميزة CI الجديدة؟

تدعم بعض أنظمة التكامل المستمر "CI"، مثل "TravisCI"، تجاهل فشل الخطوات والإبلاغ عن الوظيفة الكلية على أنها ناجحة، ولكن "CircleCI" و"GitHub Actions" لا تدعمان ذلك اعتبارًا من وقت كتابة هذه السطور.

لذلك، يمكن استخدام الحل البديل التالي:

1. `set +euo pipefail` في بداية أمر التشغيل لقمع معظم حالات الفشل المحتملة في نص أوامر "bash".
2. يجب أن يكون الأمر الأخير ناجحًا: `echo "done"` أو مجرد `true` سيؤدي الغرض.

فيما يلي مثال:

```yaml
- run:
    name: run CI experiment
    command: |
        set +euo pipefail
        echo "setting run-all-despite-any-errors-mode"
        this_command_will_fail
        echo "but bash continues to run"
        # محاكاة فشل آخر
        false
        # لكن الأمر الأخير يجب أن ينجح
        echo "خلال التجربة، لا تقم بالإزالة: الإبلاغ عن النجاح إلى CI، حتى إذا كانت هناك أخطاء"
```

بالنسبة للأوامر البسيطة، يمكنك أيضًا القيام بما يلي:

```bash
cmd_that_may_fail || true
```

بالطبع، بمجرد الرضا عن النتائج، قم بدمج الخطوة التجريبية أو المهمة مع الوظائف العادية الأخرى، مع إزالة `set +euo pipefail` أو أي أشياء أخرى قد تكون أضفتها لضمان عدم تدخل المهمة التجريبية في وظيفة CI العادية.

كانت هذه العملية برمتها ستكون أسهل بكثير إذا تمكنا فقط من تعيين شيء مثل `allow-failure` للخطوة التجريبية، والسماح لها بالفشل دون التأثير على الحالة الكلية لطلبات السحب "PRs". ولكن، كما ذكرنا سابقًا، لا تدعم "CircleCI" و"GitHub Actions" ذلك في الوقت الحالي.

يمكنك التصويت على هذه الميزة ومعرفة مكانها في هذه المواضيع الخاصة بكل نظام تكامل مستمر "CI":

- [GitHub Actions:](https://github.com/actions/toolkit/issues/399)
- [CircleCI:](https://ideas.circleci.com/ideas/CCI-I-344)

## التكامل مع DeepSpeed

بالنسبة لطلب السحب "PR" الذي يتضمن تكامل "DeepSpeed"، ضع في اعتبارك أن إعداد CI الخاص بدمج طلبات السحب "PRs" في "CircleCI" لا يحتوي على وحدات معالجة الرسوميات "GPUs". يتم تشغيل الاختبارات التي تتطلب وحدات معالجة الرسوميات "GPUs" على نظام تكامل مستمر "CI" مختلف يوميًا. وهذا يعني أنه إذا حصلت على تقرير CI ناجح في طلب السحب "PR"، فهذا لا يعني أن اختبارات "DeepSpeed" ناجحة.

لتشغيل اختبارات "DeepSpeed":

```bash
RUN_SLOW=1 pytest tests/deepspeed/test_deepspeed.py
```

يتطلب إجراء أي تغييرات على التعليم النموذجي أو رمز أمثلة "PyTorch" تشغيل اختبارات "Model Zoo" أيضًا.

```bash
RUN_SLOW=1 pytest tests/deepspeed
```