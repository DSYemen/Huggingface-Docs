# التدريب باستخدام سكريبت 

بالإضافة إلى دفاتر 🤗 Transformers [notebooks](./notebooks)، هناك أيضًا نصوص برمجية توضيحية توضح كيفية تدريب نموذج لمهمة باستخدام [PyTorch](https://github.com/huggingface/transformers/tree/main/examples/pytorch) أو [TensorFlow](https://github.com/huggingface/transformers/tree/main/examples/tensorflow) أو [JAX/Flax](https://github.com/huggingface/transformers/tree/main/examples/flax).

ستجد أيضًا نصوص برمجية استخدمها في مشاريعنا البحثية [research projects](https://github.com/huggingface/transformers/tree/main/examples/research_projects) و [أمثلة قديمة](https://github.com/huggingface/transformers/tree/main/examples/legacy) والتي ساهم بها المجتمع بشكل أساسي. هذه النصوص البرمجية غير مدعومة حاليًا وقد تتطلب إصدارًا محددًا من مكتبة 🤗 Transformers والتي من المحتمل أن تكون غير متوافقة مع أحدث إصدار من المكتبة.

لا يُتوقع أن تعمل النصوص البرمجية التوضيحية بشكل مباشر على كل مشكلة، وقد تحتاج إلى تكييف النص البرمجي مع المشكلة التي تحاول حلها. ولمساعدتك في ذلك، تعرض معظم النصوص البرمجية كيفية معالجة البيانات، مما يتيح لك تحريرها حسب الحاجة لحالتك الاستخدام.

بالنسبة لأي ميزة ترغب في تنفيذها في نص برمجي توضيحي، يرجى مناقشتها في [المنتدى](https://discuss.huggingface.co/) أو في [قضية](https://github.com/huggingface/transformers/issues) قبل إرسال طلب سحب. وفي حين أننا نرحب بإصلاح الأخطاء، فمن غير المرجح أن نقوم بدمج طلب سحب الذي يضيف المزيد من الوظائف على حساب قابلية القراءة.

سيوضح هذا الدليل كيفية تشغيل نص برمجي توضيحي للتدريب على الملخص في [PyTorch](https://github.com/huggingface/transformers/tree/main/examples/pytorch/summarization) و [TensorFlow](https://github.com/huggingface/transformers/tree/main/examples/tensorflow/summarization). ومن المتوقع أن تعمل جميع الأمثلة مع كلا الإطارين ما لم يُنص على خلاف ذلك.

## الإعداد

لتشغيل أحدث إصدار من النصوص البرمجية التوضيحية بنجاح، يجب عليك **تثبيت 🤗 Transformers من المصدر** في بيئة افتراضية جديدة:

```bash
git clone https://github.com/huggingface/transformers
cd transformers
pip install .
```

بالنسبة للإصدارات الأقدم من النصوص البرمجية التوضيحية، انقر فوق الزر أدناه:

<details>

<summary>أمثلة للإصدارات القديمة من 🤗 Transformers</summary>

<ul>

<li><a href="https://github.com/huggingface/transformers/tree/v4.5.1/examples">v4.5.1</a></li>

<li><a href="https://github.com/huggingface/transformers/tree/v4.4.2/examples">v4.4.2</a></li>

<li><a href="https://github.com/huggingface/transformers/tree/v4.3.3/examples">v4.3.3</a></li>

<li><a href="https://github.com/huggingface/transformers/tree/v4.2.2/examples">v4.2.2</a></li>

<li><a href="https://github.com/huggingface/transformers/tree/v4.1.1/examples">v4.1.1</a></li>

<li><a href="https://github.com/huggingface/transformers/tree/v4.0.1/examples">v4.0.1</a></li>

<li><a href="https://github.com/huggingface/transformers/tree/v3.5.1/examples">v3.5.1</a></li>

<li><a href="https://github.com/huggingface/transformers/tree/v3.4.0/examples">v3.4.0</a></li>

<li><a href="https://github.com/huggingface/transformers/tree/v3.3.1/examples">v3.3.1</a></li>

<li><a href="https://github.com/huggingface/transformers/tree/v3.2.0/examples">v3.2.0</a></li>

<li><a href="https://github.com/huggingface/transformers/tree/v3.1.0/examples">v3.1.0</a></li>

<li><a href="https://github.com/huggingface/transformers/tree/v3.0.2/examples">v3.0.2</a></li>

<li><a href="https://github.com/huggingface/transformers/tree/v2.11.0/examples">v2.11.0</a></li>

<li><a href="https://github.com/huggingface/transformers/tree/v2.10.0/examples">v2.10.0</a></li>

<li><a href="https://github.com/huggingface/transformers/tree/v2.9.1/examples">v2.9.1</a></li>

<li><a href="https://github.com/huggingface/transformers/tree/v2.8.0/examples">v2.8.0</a></li>

<li><a href="https://github.com/huggingface/transformers/tree/v2.7.0/examples">v2.7.0</a></li>

<li><a href="https://github.com/huggingface/transformers/tree/v2.6.0/examples">v2.6.0</a></li>

<li><a href="https://github.com/huggingface/transformers/tree/v2.5.1/examples">v2.5.1</a></li>

<li><a href="https://github.com/huggingface/transformers/tree/v2.4.0/examples">v2.4.0</a></li>

<li><a href="https://github.com/huggingface/transformers/tree/v2.3.0/examples">v2.3.0</a></li>

<li><a href="https://github.com/huggingface/transformers/tree/v2.2.0/examples">v2.2.0</a></li>

<li><a href="https://github.com/huggingface/transformers/tree/v2.1.0/examples">v2.1.1</a></li>

<li><a href="https://github.com/huggingface/transformers/tree/v2.0.0/examples">v2.0.0</a></li>

<li><a href="https://github.com/huggingface/transformers/tree/v1.2.0/examples">v1.2.0</a></li>

<li><a href="https://github.com/huggingface/transformers/tree/v1.1.0/examples">v1.1.0</a></li>

<li><a href="https://github.com/huggingface/transformers/tree/v1.0.0/examples">v1.0.0</a></li>

</ul>

</details>

ثم قم بالتبديل إلى النسخة الحالية من 🤗 Transformers إلى إصدار محدد، مثل v3.5.1 على سبيل المثال:

```bash
git checkout tags/v3.5.1
```

بعد إعداد إصدار المكتبة الصحيح، انتقل إلى مجلد الأمثلة الذي تختاره وقم بتثبيت المتطلبات الخاصة بهذا المثال:

```bash
pip install -r requirements.txt
```

## تشغيل نص برمجي

<frameworkcontent>

<pt>

يقوم النص البرمجي التوضيحي بتنزيل ومعالجة مجموعة بيانات من مكتبة 🤗 [Datasets](https://huggingface.co/docs/datasets/). ثم يقوم النص البرمجي بضبط نموذج باستخدام [Trainer](https://huggingface.co/docs/transformers/main_classes/trainer) على بنية تدعم الملخص. يوضح المثال التالي كيفية ضبط نموذج [T5-small](https://huggingface.co/google-t5/t5-small) على مجموعة بيانات [CNN/DailyMail](https://huggingface.co/datasets/cnn_dailymail). يتطلب نموذج T5 حجة `source_prefix` إضافية بسبب الطريقة التي تم تدريبه بها. يعمل هذا المطالبة على إعلام T5 بأن هذه مهمة ملخص.

```bash
python examples/pytorch/summarization/run_summarization.py \
--model_name_or_path google-t5/t5-small \
--do_train \
--do_eval \
--dataset_name cnn_dailymail \
--dataset_config "3.0.0" \
--source_prefix "summarize: " \
--output_dir /tmp/tst-summarization \
--per_device_train_batch_size=4 \
--per_device_eval_batch_size=4 \
--overwrite_output_dir \
--predict_with_generate
```

</pt>

<tf>

يقوم النص البرمجي التوضيحي بتنزيل ومعالجة مجموعة بيانات من مكتبة 🤗 [Datasets](https://huggingface.co/docs/datasets/). ثم يقوم النص البرمجي بضبط نموذج باستخدام Keras على بنية تدعم الملخص. يوضح المثال التالي كيفية ضبط نموذج [T5-small](https://huggingface.co/google-t5/t5-small) على مجموعة بيانات [CNN/DailyMail](https://huggingface.co/datasets/cnn_dailymail). يتطلب نموذج T5 حجة `source_prefix` إضافية بسبب الطريقة التي تم تدريبه بها. يعمل هذا المطالبة على إعلام T5 بأن هذه مهمة ملخص.

```bash
python examples/tensorflow/summarization/run_summarization.py  \
--model_name_or_path google-t5/t5-small \
--dataset_name cnn_dailymail \
--dataset_config "3.0.0" \
--output_dir /tmp/tst-summarization  \
--per_device_train_batch_size 8 \
--per_device_eval_batch_size 16 \
--num_train_epochs 3 \
--do_train \
--do_eval
```

</tf>

</frameworkcontent>

## التدريب الموزع والدقة المختلطة

يدعم [Trainer](https://huggingface.co/docs/transformers/main_classes/trainer) التدريب الموزع والدقة المختلطة، مما يعني أنه يمكنك أيضًا استخدامه في نص برمجي. لتمكين كلتا الميزتين:

- أضف حجة `fp16` لتمكين الدقة المختلطة.

- قم بتعيين عدد وحدات معالجة الرسومات (GPUs) التي تريد استخدامها باستخدام حجة `nproc_per_node`.

```bash
torchrun \
--nproc_per_node 8 pytorch/summarization/run_summarization.py \
--fp16 \
--model_name_or_path google-t5/t5-small \
--do_train \
--do_eval \
--dataset_name cnn_dailymail \
--dataset_config "3.0.0" \
--source_prefix "summarize: " \
--output_dir /tmp/tst-summarization \
--per_device_train_batch_size=4 \
--per_device_eval_batch_size=4 \
--overwrite_output_dir \
--predict_with_generate
```

تستخدم نصوص TensorFlow البرمجية استراتيجية [`MirroredStrategy`](https://www.tensorflow.org/guide/distributed_training#mirroredstrategy) للتدريب الموزع، ولا تحتاج إلى إضافة أي حجج إضافية إلى نص التدريب البرمجي. سيستخدم نص TensorFlow البرمجي بشكل افتراضي وحدات معالجة الرسومات (GPUs) متعددة إذا كانت متوفرة.
## تشغيل برنامج نصي على وحدة معالجة Tensor 

تم تصميم وحدات معالجة Tensor (TPUs) خصيصًا لتسريع الأداء. يدعم باي تورش TPUs مع مجمع XLA للتعلم العميق (راجع هنا لمزيد من التفاصيل). لاستخدام وحدة TPU، قم بتشغيل برنامج xla_spawn.py النصي واستخدم وسيطة num_cores لتحديد عدد أنوية TPU التي تريد استخدامها.

## تشغيل برنامج نصي باستخدام Accelerate 

Accelerate هي مكتبة باي تورش فقط توفر طريقة موحدة لتدريب نموذج على عدة أنواع من الإعدادات (CPU فقط، وGPUs متعددة، ووحدات TPU) مع الحفاظ على الرؤية الكاملة لحلقة تدريب باي تورش. تأكد من تثبيت Accelerate إذا لم يكن لديك بالفعل:

> ملاحظة: نظرًا لأن Accelerate في مرحلة تطوير سريعة، يجب تثبيت إصدار Git من Accelerate لتشغيل البرامج النصية.

بدلاً من برنامج run_summarization.py النصي، تحتاج إلى استخدام برنامج run_summarization_no_trainer.py النصي. ستكون البرامج النصية المدعومة من Accelerate لها ملف task_no_trainer.py في المجلد. ابدأ بتشغيل الأمر التالي لإنشاء وحفظ ملف تكوين:

اختبر إعدادك للتأكد من تهيئته بشكل صحيح:

الآن أنت مستعد لبدء التدريب:

## استخدام مجموعة بيانات مخصصة 

يدعم برنامج النص البرمجي للتلخيص مجموعات البيانات المخصصة طالما أنها ملف CSV أو JSON Line. عند استخدام مجموعة البيانات الخاصة بك، يلزمك تحديد العديد من الحجج الإضافية:

- train_file وvalidation_file تحدد مسار ملفات التدريب والتحقق الخاصة بك.
- text_column هو النص المدخل الذي سيتم تلخيصه.
- summary_column هو النص المستهدف الذي سيتم إخراجه.

سيكون مظهر برنامج النص البرمجي للتلخيص الذي يستخدم مجموعة بيانات مخصصة على النحو التالي:

## اختبار برنامج نصي 

غالبًا ما يكون من الجيد تشغيل البرنامج النصي الخاص بك على عدد أقل من أمثلة مجموعة البيانات للتأكد من أن كل شيء يعمل كما هو متوقع قبل الالتزام بمجموعة بيانات كاملة قد تستغرق ساعات لإكمالها. استخدم الحجج التالية لتقليص مجموعة البيانات إلى عدد أقصى من العينات:

- max_train_samples
- max_eval_samples
- max_predict_samples

لا تدعم جميع البرامج النصية للمثال حجة max_predict_samples. إذا لم تكن متأكدًا مما إذا كان برنامجك النصي يدعم هذه الحجة، فأضف حجة -h للتحقق:

## استئناف التدريب من نقطة تفتيش 

خيار آخر مفيد لتمكينه هو استئناف التدريب من نقطة تفتيش سابقة. سيضمن ذلك إمكانية الاستمرار من حيث توقفت دون البدء من جديد إذا تم مقاطعة تدريبك. هناك طريقتان لاستئناف التدريب من نقطة تفتيش.

تستخدم الطريقة الأولى حجة output_dir previous_output_dir لاستئناف التدريب من أحدث نقطة تفتيش مخزنة في output_dir. في هذه الحالة، يجب إزالة "overwrite_output_dir":

تستخدم الطريقة الثانية حجة resume_from_checkpoint path_to_specific_checkpoint لاستئناف التدريب من مجلد نقطة تفتيش محددة.

## مشاركة نموذجك 

يمكن لجميع البرامج النصية تحميل نموذجك النهائي إلى Model Hub. تأكد من تسجيل الدخول إلى Hugging Face قبل البدء:

ثم أضف حجة push_to_hub إلى البرنامج النصي. سيؤدي هذا الحجة إلى إنشاء مستودع باسم مستخدم Hugging Face الخاص بك واسم المجلد المحدد في output_dir.

لإعطاء مستودعك اسمًا محددًا، استخدم حجة push_to_hub_model_id لإضافته. سيتم سرد المستودع تلقائيًا ضمن مساحة الاسم الخاصة بك.

يوضح المثال التالي كيفية تحميل نموذج باسم مستودع محدد: