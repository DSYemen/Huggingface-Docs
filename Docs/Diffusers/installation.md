# التثبيت 

🤗 Diffusers تم اختباره على Python 3.8+، و PyTorch 1.7.0+، و Flax. اتبع تعليمات التثبيت أدناه لمكتبة التعلم العميق التي تستخدمها:

- تعليمات تثبيت [PyTorch](https://pytorch.org/get-started/locally/)
- تعليمات تثبيت [Flax](https://flax.readthedocs.io/en/latest/)

## التثبيت باستخدام pip

يجب تثبيت 🤗 Diffusers في [بيئة افتراضية](https://docs.python.org/3/library/venv.html). إذا لم تكن معتادًا على البيئات الافتراضية في Python، فراجع هذا [الدليل](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/). تجعل البيئة الافتراضية من السهل إدارة المشاريع المختلفة وتجنب مشكلات التوافق بين التبعيات.

ابدأ بإنشاء بيئة افتراضية في دليل مشروعك:

```bash
python -m venv .env
```

قم بتنشيط البيئة الافتراضية:

```bash
source .env/bin/activate
```

يجب عليك أيضًا تثبيت 🤗 Transformers لأن 🤗 Diffusers يعتمد على نماذجه:

<frameworkcontent>
<pt>

ملاحظة - تدعم PyTorch فقط Python 3.8 - 3.11 على Windows.

```bash
pip install diffusers["torch"] transformers
```

</pt>
<jax>

```bash
pip install diffusers["flax"] transformers
```

</jax>
</frameworkcontent>

## التثبيت باستخدام conda

بعد تنشيط بيئتك الافتراضية، مع `conda` (مدعوم من المجتمع):

```bash
conda install -c conda-forge diffusers
```

## التثبيت من المصدر

قبل تثبيت 🤗 Diffusers من المصدر، تأكد من تثبيت PyTorch و 🤗 Accelerate.

لتثبيت 🤗 Accelerate:

```bash
pip install accelerate
```

ثم قم بتثبيت 🤗 Diffusers من المصدر:

```bash
pip install git+https://github.com/huggingface/diffusers
```

هذه الأوامر تقوم بتثبيت نسخة "main" النزيف الحافة بدلا من أحدث إصدار "مستقر".

يعد إصدار "main" مفيدًا للبقاء على اطلاع بأحدث التطورات. على سبيل المثال، إذا تم إصلاح خطأ منذ الإصدار الرسمي الأخير ولكن لم يتم طرح إصدار جديد بعد. ومع ذلك، فإن هذا يعني أن إصدار "main" قد لا يكون مستقرًا دائمًا.

نحن نسعى جاهدين للحفاظ على إصدار "main" قيد التشغيل، ويتم حل معظم المشكلات عادةً في غضون بضع ساعات أو يوم. إذا واجهتك مشكلة، يرجى فتح [قضية](https://github.com/huggingface/diffusers/issues/new/choose) حتى نتمكن من إصلاحها في أقرب وقت ممكن!

## التثبيت القابل للتحرير

ستحتاج إلى تثبيت قابل للتحرير إذا كنت ترغب في:

* استخدام إصدار "main" من كود المصدر.
* المساهمة في 🤗 Diffusers وتحتاج إلى اختبار التغييرات في الكود.

قم باستنساخ المستودع وقم بتثبيت 🤗 Diffusers بالأوامر التالية:

```bash
git clone https://github.com/huggingface/diffusers.git
cd diffusers
```

<frameworkcontent>
<pt>

```bash
pip install -e ".[torch]"
```

</pt>
<jax>

```bash
pip install -e ".[flax]"
```

</jax>
</frameworkcontent>

ستقوم هذه الأوامر بربط المجلد الذي استنسخت منه المستودع ومسارات مكتبة Python.

سيبحث Python الآن داخل المجلد الذي قمت باستنساخه بالإضافة إلى مسارات المكتبة العادية. على سبيل المثال، إذا تم تثبيت حزم Python الخاصة بك عادةً في `~/anaconda3/envs/main/lib/python3.10/site-packages/`, فسيقوم Python أيضًا بالبحث في مجلد `~/diffusers/` الذي استنسخته إليه.

<Tip warning={true}>

يجب عليك الاحتفاظ بمجلد "diffusers" إذا كنت تريد الاستمرار في استخدام المكتبة.

</Tip>

الآن يمكنك تحديث المستنسخ الخاص بك بسهولة إلى أحدث إصدار من 🤗 Diffusers بالأمر التالي:

```bash
cd ~/diffusers/
git pull
```

ستجد بيئة Python إصدار "main" من 🤗 Diffusers في المرة التالية التي يتم تشغيلها فيها.

## ذاكرة التخزين المؤقت

يتم تنزيل أوزان النموذج والملفات من Hub إلى ذاكرة التخزين المؤقت التي تكون عادةً دليل المنزل الخاص بك. يمكنك تغيير موقع ذاكرة التخزين المؤقت عن طريق تحديد متغيرات البيئة `HF_HOME` أو `HUGGINFACE_HUB_CACHE` أو تكوين معلمة `cache_dir` في طرق مثل [`~DiffusionPipeline.from_pretrained`].

تسمح الملفات المخزنة مؤقتًا بتشغيل 🤗 Diffusers دون اتصال بالإنترنت. لمنع 🤗 Diffusers من الاتصال بالإنترنت، قم بتعيين متغير البيئة `HF_HUB_OFFLINE` على `True` ولن يقوم 🤗 Diffusers بتحميل سوى الملفات التي تم تنزيلها مسبقًا في ذاكرة التخزين المؤقت.

```shell
export HF_HUB_OFFLINE=True
```

للحصول على مزيد من التفاصيل حول إدارة وتنظيف ذاكرة التخزين المؤقت، راجع دليل [التخزين المؤقت](https://huggingface.co/docs/huggingface_hub/guides/manage-cache).

## تسجيل بيانات الاستخدام

تجمع مكتبتنا معلومات الاستخدام عن بُعد أثناء طلبات [`~DiffusionPipeline.from_pretrained`].

تشمل البيانات المجمعة إصدار 🤗 Diffusers و PyTorch/Flax، وطلب فئة النموذج أو الأنابيب، ومسار نقطة التحقق المسبقة التدريب إذا تم استضافتها على Hugging Face Hub.

تساعدنا بيانات الاستخدام هذه على تصحيح الأخطاء وتحديد أولويات الميزات الجديدة.

يتم إرسال بيانات الاستخدام فقط عند تحميل النماذج والأنابيب من Hub، ولا يتم جمعها إذا كنت تقوم بتحميل ملفات محلية.

نحن ندرك أن الجميع لا يريدون مشاركة معلومات إضافية، ونحن نحترم خصوصيتك. يمكنك تعطيل جمع بيانات الاستخدام عن طريق تعيين متغير البيئة `DISABLE_TELEMETRY` من المحطة الطرفية الخاصة بك:

على Linux/MacOS:

```bash
export DISABLE_TELEMETRY=YES
```

على Windows:

```bash
set DISABLE_TELEMETRY=YES
```