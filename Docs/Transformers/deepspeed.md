# DeepSpeed

مكتبة PyTorch للتحسين تجعل التدريب الموزع فعال من حيث الذاكرة وسريع. وفي جوهره [مُحسن Zero Redundancy (ZeRO)](https://hf.co/papers/1910.02054) الذي يتيح تدريب نماذج كبيرة على نطاق واسع. يعمل ZeRO على عدة مراحل:

* ZeRO-1، تجزئة حالة المحسن عبر وحدات معالجة الرسوميات (GPU)
* ZeRO-2، تجزئة التدرج عبر وحدات معالجة الرسوميات (GPU)
* ZeRO-3، تجزئة المعلمات عبر وحدات معالجة الرسوميات (GPU)

في البيئات المحدودة بوحدة معالجة الرسوميات (GPU)، يمكّن ZeRO أيضًا من نقل ذاكرة المحسن والحساب من وحدة معالجة الرسوميات (GPU) إلى وحدة المعالجة المركزية (CPU) لتناسب وتدريب النماذج الكبيرة حقًا على وحدة معالجة الرسوميات (GPU) واحدة. تم دمج DeepSpeed مع فئة محولات [`Trainer`] لجميع مراحل ZeRO والنقل. كل ما عليك فعله هو توفير ملف تكوين أو يمكنك استخدام قالب مقدم. للتنبؤ، تدعم المحولات ZeRO-3 والنقل لأنه يسمح بتحميل نماذج ضخمة.

سيوضح لك هذا الدليل كيفية نشر تدريب DeepSpeed والميزات التي يمكنك تمكينها وكيفية إعداد ملفات التكوين لمراحل ZeRO المختلفة والنقل والاستدلال واستخدام DeepSpeed بدون [`Trainer`].

## التثبيت

DeepSpeed متاح لتثبيته من PyPI أو المحولات (لمزيد من خيارات التثبيت التفصيلية، راجع تفاصيل تثبيت DeepSpeed [التفاصيل](https://www.deepspeed.ai/tutorials/advanced-install/) أو [README](https://github.com/microsoft/deepspeed#installation) على GitHub).

<Tip>

إذا كنت تواجه صعوبة في تثبيت DeepSpeed، فراجع دليل تثبيت CUDA DeepSpeed [](../debugging#deepspeed-cuda-installation). على الرغم من أن DeepSpeed يحتوي على حزمة PyPI قابلة للتثبيت باستخدام pip، إلا أنه يوصى بشدة [بتثبيته من المصدر](https://www.deepspeed.ai/tutorials/advanced-install/#install-deepspeed-from-source) لمطابقة أجهزتك بشكل أفضل ولدعم ميزات معينة، مثل 1-bit Adam، والتي ليست متوفرة في توزيع PyPI.

</Tip>

<hfoptions id="install">

<hfoption id="PyPI">

```bash
pip install deepspeed
```

</hfoption>

<hfoption id="Transformers">

```bash
pip install transformers[deepspeed]
```

</hfoption>

</hfoptions>

## متطلبات الذاكرة

قبل البدء، من الجيد التحقق مما إذا كان لديك ذاكرة GPU وCPU كافية لتناسب نموذجك. توفر DeepSpeed أداة لتقدير ذاكرة CPU/GPU المطلوبة. على سبيل المثال، لتقدير متطلبات الذاكرة لنموذج [bigscience/T0_3B](bigscience/T0_3B) على وحدة معالجة الرسوميات (GPU) واحدة:

```bash
$ python -c 'from transformers import AutoModel; \
> from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live; \
> model = AutoModel.from_pretrained("bigscience/T0_3B"); \
> estimate_zero3_model_states_mem_needs_all_live(model, num_gpus_per_node=1, num_nodes=1)'
[...]
Estimated memory needed for params, optim states and gradients for a:
HW: Setup with 1 node, 1 GPU per node.
SW: Model with 2783M total params, 65M largest layer params.
per CPU  |  per GPU |   Options
70.00GB |   0.25GB | offload_param=cpu , offload_optimizer=cpu , zero_init=1
70.0Multiplier |   0.25GB | offload_param=cpu , offload_optimizer=cpu , zero_init=0
62.23GB |   5.43GB | offload_param=none, offload_optimizer=cpu , zero_init=1
62.23GB |   5.43GB | offload_param=none, offload_optimizer=cpu , zero_init=0
0.37GB |  46.91GB | offload_param=none, offload_optimizer=none, zero_init=1
15.56GB |  46.91GB | offload_param=none, offload_optimizer=none, zero_init=0
```

هذا يعني أنك إما بحاجة إلى وحدة معالجة رسومات (GPU) واحدة بسعة 80 جيجابايت بدون نقل إلى وحدة المعالجة المركزية (CPU) أو وحدة معالجة رسومات (GPU) بسعة 8 جيجابايت ووحدة معالجة مركزية (CPU) بسعة 60 جيجابايت لنقلها (هذه هي متطلبات الذاكرة للمعلمات وحالات المحسنات والتدرجات فقط، وستحتاج إلى المزيد قليلاً لنواة CUDA والتنشيط). يجب عليك أيضًا مراعاة المقايضة بين التكلفة والسرعة لأنه سيكون من الأرخص استئجار أو شراء وحدة معالجة رسومات (GPU) أصغر ولكن سيستغرق تدريب نموذجك وقتًا أطول.

إذا كانت لديك ذاكرة GPU كافية، فتأكد من تعطيل النقل من CPU/NVMe لجعل كل شيء أسرع.

## حدد مرحلة ZeRO

بعد تثبيت DeepSpeed ومعرفة متطلبات الذاكرة الخاصة بك، تتمثل الخطوة التالية في تحديد مرحلة ZeRO لاستخدامها. حسب الترتيب الأسرع والأكثر كفاءة في الذاكرة:

| الأسرع          | الأكثر كفاءة في الذاكرة |
|------------------|------------------|
| ZeRO-1           | ZeRO-3 + النقل   |
| ZeRO-2           | ZeRO-3           |
| ZeRO-2 + النقل | ZeRO-2 + النقل |
| ZeRO-3           | ZeRO-2           |
| ZeRO-3 + النقل | ZeRO-1           |

لمعرفة ما يناسبك، ابدأ بالنهج الأسرع وإذا نفدت الذاكرة، فجرّب المرحلة التالية التي تكون أبطأ ولكنها أكثر كفاءة في الذاكرة. لا تتردد في العمل في أي اتجاه تفضله (بدءًا من الأكثر كفاءة في الذاكرة أو الأسرع) لاكتشاف التوازن المناسب بين السرعة واستخدام الذاكرة.

يمكنك استخدام عملية عامة (ابدأ بحجم دفعة يبلغ 1):

1. تمكين نقاط تفتيش التدرج
2. جرب ZeRO-2
3. جرب ZeRO-2 ونقل المحسن
4. جرب ZeRO-3
5. جرب ZeRO-3 ونقل المعلمات إلى وحدة المعالجة المركزية (CPU)
6. جرب ZeRO-3 ونقل المعلمات والمحسن إلى وحدة المعالجة المركزية (CPU)
7. جرب خفض القيم الافتراضية المختلفة مثل شعاع بحث أضيق إذا كنت تستخدم طريقة [`~GenerationMixin.generate`]
8. جرب الدقة المختلطة نصفية الدقة (fp16 على معماريات GPU الأقدم وbf16 على Ampere) على الأوزان ذات الدقة الكاملة
9. أضف المزيد من الأجهزة إذا أمكن ذلك أو قم بتمكين Infinity لنقل المعلمات والمحسن إلى NVMe
10. بمجرد عدم نفاد الذاكرة، قم بقياس الإنتاجية الفعالة ثم حاول زيادة حجم الدفعة قدر الإمكان لتعظيم كفاءة وحدة معالجة الرسوميات (GPU)
11. أخيرًا، حاول تحسين إعداد التدريب الخاص بك عن طريق تعطيل بعض ميزات النقل أو استخدام مرحلة ZeRO أسرع وزيادة/إنقاص حجم الدفعة للعثور على أفضل مقايضة بين السرعة واستخدام الذاكرة

## ملف تكوين DeepSpeed

يعمل DeepSpeed مع فئة [`Trainer`] من خلال ملف تكوين يحتوي على جميع المعلمات لتكوين كيفية إعداد تشغيل التدريب الخاص بك. عندما تقوم بتنفيذ نص برمجي للتدريب، يقوم DeepSpeed بتسجيل التكوين الذي تلقاه من [`Trainer`] في وحدة التحكم حتى تتمكن من رؤية التكوين المستخدم بالضبط.

<Tip>

يمكنك العثور على قائمة كاملة بخيارات تكوين DeepSpeed في مرجع تكوين DeepSpeed JSON [](https://www.deepspeed.ai/docs/config-json/). يمكنك أيضًا العثور على أمثلة أكثر عملية لمختلف أمثلة تكوين DeepSpeed على مستودع [DeepSpeedExamples](https://github.com/microsoft/DeepSpeedExamples) أو المستودع الرئيسي [DeepSpeed](https://github.com/microsoft/DeepSpeed). للعثور بسرعة على أمثلة محددة، يمكنك:

```bash
git clone https://github.com/microsoft/DeepSpeedExamples
cd DeepSpeedExamples
find . -name '*json'
# find examples with the Lamb optimizer
grep -i Lamb $(find . -name '*json')
```

</Tip>

يتم تمرير ملف تكوين DeepSpeed كمسار إلى ملف JSON إذا كنت تقوم بالتدريب من واجهة سطر الأوامر أو ككائن `dict` متداخل إذا كنت تستخدم [`Trainer`] في إعداد دفتر الملاحظات.

<hfoptions id="pass-config">

<hfoption id="path to file">

```py
TrainingArguments(..., deepspeed="path/to/deepspeed_config.json")
```

</hfoption>

<hfoption id="nested dict">

```py
ds_config_dict = dict(scheduler=scheduler_params, optimizer=optimizer_params)
args = TrainingArguments(..., deepspeed=ds_config_dict)
trainer = Trainer(model, args, ...)
```

</hfoption>

</hfoptions>

### معلمات DeepSpeed وTrainer

هناك ثلاثة أنواع من معلمات التكوين:

1. بعض معلمات التكوين مشتركة بين [`Trainer`] وDeepSpeed، ويمكن أن يكون من الصعب تحديد الأخطاء عندما تكون هناك تعريفات متضاربة. لتسهيل الأمر، يتم تكوين هذه المعلمات المشتركة من وسيطات سطر الأوامر [`Trainer`].

2. يتم اشتقاق بعض معلمات التكوين تلقائيًا من تكوين النموذج، لذلك لا تحتاج إلى ضبط هذه القيم يدويًا. يستخدم [`Trainer`] قيمة التكوين `auto` لتحديد الإعداد الأكثر صحة أو كفاءة. يمكنك تعيين معلمات التكوين الخاصة بك بشكل صريح، ولكن يجب عليك التأكد من اتفاق وسيطات [`Trainer`] ومعلمات تكوين DeepSpeed. قد تسبب عدم التطابقات فشل التدريب بطرق يصعب اكتشافها!

3. بعض معلمات التكوين الخاصة بـ DeepSpeed فقط والتي تحتاج إلى إعداد يدويًا بناءً على احتياجات التدريب الخاصة بك.

يمكنك أيضًا تعديل تكوين DeepSpeed وتحرير [`TrainingArguments`] منه:

1. قم بإنشاء أو تحميل تكوين DeepSpeed لاستخدامه كتكوين أساسي
2. قم بإنشاء كائن [`TrainingArguments`] بناءً على قيم تكوين DeepSpeed هذه

يقوم [`Trainer`] بحساب بعض القيم، مثل `scheduler.params.total_num_steps`، أثناء التدريب.
### تكوين ZeRO

هناك ثلاثة تكوينات، لكل منها مرحلة ZeRO مختلفة. المرحلة 1 ليست مثيرة للاهتمام من حيث قابلية التوسع، ويركز هذا الدليل على المرحلتين 2 و3. يحتوي تكوين "zero_optimization" على جميع الخيارات لما يجب تمكينه وكيفية تكوينه. للحصول على شرح أكثر تفصيلاً لكل معلمة، راجع مرجع [تكوين DeepSpeed JSON](https://www.deepspeed.ai/docs/config-json/).

> <Tip warning={true}>
> لا يتحقق DeepSpeed من أسماء المعلمات وأي أخطاء إملائية تعود إلى الإعداد الافتراضي للمعلمة. يمكنك مراقبة رسائل سجل بدء تشغيل DeepSpeed Engine لمعرفة القيم التي سيتم استخدامها.
> </Tip>

يجب إعداد التكوينات التالية باستخدام DeepSpeed لأن [`Trainer`] لا يوفر حجج سطر الأوامر المكافئة.

<hfoptions id="zero-config">
<hfoption id="ZeRO-1">

يقسم ZeRO-1 حالات المحسن عبر وحدات معالجة الرسوميات (GPU)، ويمكنك توقع تسريع بسيط. يمكن إعداد تكوين ZeRO-1 على النحو التالي:

```yml
{
"zero_optimization": {
"stage": 1
}
}
```

</hfoption>

<hfoption id="ZeRO-2">

يقسم ZeRO-2 المحسن والتدرجات عبر وحدات معالجة الرسوميات (GPU). تُستخدم هذه المرحلة بشكل أساسي للتدريب لأن ميزاتها غير ذات صلة بالاستدلال. بعض المعلمات المهمة التي يجب تكوينها لتحقيق أداء أفضل تشمل ما يلي:

- `offload_optimizer` يجب تمكينه لتقليل استخدام ذاكرة GPU.
- `overlap_comm` عند تعيينه إلى `true`، فإنه يقلل من استخدام ذاكرة GPU لخفض تأخير allreduce. تستخدم هذه الميزة 4.5x قيم `allgather_bucket_size` و`reduce_bucket_size`. في هذا المثال، يتم تعيينها على `5e8`، مما يعني أنها تتطلب 9 جيجابايت من ذاكرة GPU. إذا كانت ذاكرة GPU لديك 8 جيجابايت أو أقل، فيجب عليك تقليل `overlap_comm` لتقليل متطلبات الذاكرة ومنع حدوث خطأ في الذاكرة (OOM).
- `allgather_bucket_size` و`reduce_bucket_size` يوفران ذاكرة GPU المتاحة مقابل سرعة الاتصال. كلما صغرت قيمهما، تباطأت الاتصالات وأصبحت ذاكرة GPU المتاحة أكبر. يمكنك الموازنة، على سبيل المثال، بين ما إذا كان حجم الدفعة الأكبر أكثر أهمية من وقت التدريب البطيء قليلاً.
- `round_robin_gradients` متاح في DeepSpeed 0.4.4 لتفريغ الذاكرة المؤقتة لـ CPU. فهو يوازي نسخ التدرج إلى ذاكرة CPU بين الترتيبات عن طريق تقسيم التدرج الدقيق. تنمو الفائدة في الأداء مع خطوات تجميع التدرج (مزيد من النسخ بين خطوات المحسن) أو عدد وحدات معالجة الرسوميات (زيادة الموازية).

```yml
{
"zero_optimization": {
"stage": 2,
"offload_optimizer": {
"device": "cpu",
"pin_memory": true
},
"allgather_partitions": true,
"allgather_bucket_size": 5e8,
"overlap_comm": true,
"reduce_scatter": true,
"reduce_bucket_size": 5e8,
"contiguous_gradients": true
"round_robin_gradients": true
}
}
```

</hfoption>

<hfoption id="ZeRO-3">

يقسم ZeRO-3 المحسن والتدرجات والمعلمات عبر وحدات معالجة الرسوميات (GPU). على عكس ZeRO-2، يمكن أيضًا استخدام ZeRO-3 للاستدلال، بالإضافة إلى التدريب، لأنه يسمح بتحميل نماذج كبيرة على وحدات معالجة الرسوميات (GPU) متعددة. بعض المعلمات المهمة التي يجب تكوينها تشمل ما يلي:

- `device: "cpu"` يمكن أن يساعد إذا كنت تواجه مشكلة في نفاد ذاكرة GPU، وإذا كانت لديك ذاكرة CPU متوفرة. يسمح ذلك بتفريغ معلمات النموذج إلى وحدة المعالجة المركزية (CPU).
- `pin_memory: true` يمكن أن يحسن الإنتاجية، ولكن تصبح ذاكرة أقل متاحة للعمليات الأخرى لأن الذاكرة المثبتة محجوزة للعملية المحددة التي طلبتها ويتم الوصول إليها عادة بشكل أسرع من ذاكرة CPU العادية.
- `stage3_max_live_parameters` هو الحد الأعلى لعدد المعلمات الكاملة التي تريد الاحتفاظ بها في وحدة معالجة الرسوميات (GPU) في أي وقت. قلل هذه القيمة إذا واجهت خطأ في الذاكرة (OOM).
- `stage3_max_reuse_distance` هي قيمة لتحديد ما إذا كانت المعلمة ستستخدم مرة أخرى في المستقبل، وتساعد في اتخاذ قرار بشأن ما إذا كان سيتم التخلص من المعلمة أو الاحتفاظ بها. إذا كانت المعلمة ستُستخدم مرة أخرى (إذا كانت القيمة أقل من `stage3_max_reuse_distance`)، فسيتم الاحتفاظ بها لتقليل النفقات العامة للاتصال. هذا مفيد للغاية عندما يتم تمكين نقطة تفتيش التنشيط وتريد الاحتفاظ بالمعلمة في إعادة الحساب للأمام حتى المرور الخلفي. ولكن قلل هذه القيمة إذا واجهت خطأ في الذاكرة (OOM).
- `stage3_gather_16bit_weights_on_model_save` توحيد أوزان fp16 عند حفظ نموذج. بالنسبة للنماذج الكبيرة ومتعددة وحدات معالجة الرسوميات (GPU)، فإن هذا مكلف من حيث الذاكرة والسرعة. يجب تمكينه إذا كنت تخطط لاستئناف التدريب.
- `sub_group_size` يتحكم في المعلمات التي يتم تحديثها أثناء خطوة المحسن. يتم تجميع المعلمات في دلوات من `sub_group_size` ويتم تحديث كل دلو في وقت واحد. عند استخدامه مع تفريغ NVMe، يحدد `sub_group_size` متى يتم نقل حالات النموذج من وإلى ذاكرة CPU أثناء خطوة التحسين. يمنع ذلك نفاد ذاكرة CPU للنطاقات الكبيرة للغاية. يمكن ترك `sub_group_size` بقيمته الافتراضية إذا لم تكن تستخدم تفريغ NVMe، ولكن قد ترغب في تغييره إذا:

1. واجهت خطأ في الذاكرة (OOM) أثناء خطوة المحسن. في هذه الحالة، قلل `sub_group_size` لتقليل استخدام الذاكرة المؤقتة.
2. تستغرق خطوة المحسن وقتًا طويلاً. في هذه الحالة، قم بزيادة `sub_group_size` لتحسين استخدام النطاق الترددي نتيجة زيادة مخازن البيانات.

- `reduce_bucket_size`، و`stage3_prefetch_bucket_size`، و`stage3_param_persistence_threshold` تعتمد على الحجم المخفي للنموذج. يوصى بتعيين هذه القيم إلى "auto" والسماح لـ [`Trainer`] بتعيين القيم تلقائيًا.

```yml
{
"zero_optimization": {
"stage": 3,
"offload_optimizer": {
"device": "cpu",
"pin_memory": true
},
"offload_param": {
"device": "cpu",
"pin_memory": true
},
"overlap_comm": true,
"contiguous_gradients": true,
"sub_group_size": 1e9,
"reduce_bucket_size": "auto",
"stage3_prefetch_bucket_size": "auto",
"stage3_param_persistence_threshold": "auto",
"stage3_max_live_parameters": 1e9,
"stage3_max_reuse_distance": 1e9,
"stage3_gather_16bit_weights_on_model_save": true
}
}
```

يمكنك استخدام [`deepspeed.zero.Init`](https://deepspeed.readthedocs.io/en/latest/zero3.html#deepspeed.zero.Init) context manager لتهيئة نموذج بشكل أسرع:

```py
from transformers import T5ForConditionalGeneration, T5Config
import deepspeed

with deepspeed.zero.Init():
config = T5Config.from_pretrained("google-t5/t5-small")
model = T5ForConditionalGeneration(config)
```

بالنسبة للنماذج المُدربة مسبقًا، يجب أن يحتوي ملف تكوين DeepSped على `is_deepspeed_zero3_enabled: true` المُعد في [`TrainingArguments`]، ويجب تمكين تكوين ZeRO. يجب إنشاء كائن [`TrainingArguments`] **قبل** استدعاء نموذج [`~PreTrainedModel.from_pretrained`].

```py
from transformers import AutoModel، Trainer، TrainingArguments

training_args = TrainingArguments(..., deepspeed=ds_config)
model = AutoModel.from_pretrained("google-t5/t5-small")
trainer = Trainer(model=model، args=training_args، ...)
```

ستحتاج إلى ZeRO-3 إذا لم تتسع أوزان fp16 على وحدة معالجة الرسوميات (GPU) واحدة. إذا كنت قادرًا على تحميل أوزان fp16، فتأكد من تحديد `torch_dtype=torch.float16` في [`~PreTrainedModel.from_pretrained`].

اعتبار آخر لـ ZeRO-3 هو إذا كان لديك وحدات معالجة رسومات متعددة، فلا تحتوي وحدة معالجة الرسوميات (GPU) واحدة على جميع المعلمات ما لم تكن معلمات للطبقة التي يتم تنفيذها حاليًا. للوصول إلى جميع المعلمات من جميع الطبقات في وقت واحد، مثل تحميل أوزان النموذج المُدربة مسبقًا في [`~PreTrainedModel.from_pretrained`]]، يتم تحميل طبقة واحدة في كل مرة وتقسيمها على الفور على جميع وحدات معالجة الرسوميات (GPU). ويرجع ذلك إلى أنه بالنسبة للنماذج الكبيرة جدًا، لا يمكن تحميل الأوزان على وحدة معالجة الرسوميات (GPU) واحدة ثم توزيعها عبر وحدات معالجة الرسوميات (GPU) الأخرى بسبب قيود الذاكرة.

إذا صادفت وزن معلمة نموذج مثل التالي، حيث `tensor([1.])` أو حجم المعلمة هو 1 بدلاً من شكل متعدد الأبعاد أكبر، فهذا يعني أن المعلمة مقسمة وهذا هو عنصر نائب ZeRO-3.

```py
tensor([1.0]، device="cuda:0"، dtype=torch.float16، requires_grad=True)
```

<Tip>

للحصول على مزيد من المعلومات حول تهيئة النماذج الكبيرة باستخدام ZeRO-3 والوصول إلى المعلمات، راجع دليلي [بناء النماذج الضخمة](https://deepspeed.readthedocs.io/en/latest/zero3.html#constructing-massive-models) و[جمع المعلمات](https://deepspeed.readthedocs.io/en/latest/zero3.html#gathering-parameters).

</Tip>

</hfoption>
</hfoptions>
### تهيئة NVMe

تتيح [ZeRO-Infinity](https://hf.co/papers/2104.07857) إمكانية تفريغ حالات النموذج إلى وحدة المعالجة المركزية و/أو NVMe لتوفير المزيد من الذاكرة. تسمح خوارزميات التجزئة والتبليط الذكية لكل وحدة معالجة رسومية بإرسال واستقبال كميات صغيرة جدًا من البيانات أثناء التفريغ بحيث يمكن لجهاز NVMe الحديث أن يتناسب مع مجموعة ذاكرة إجمالية أكبر من المتوفرة لعملية التدريب الخاصة بك. تتطلب ZeRO-Infinity ZeRO-3.

اعتمادًا على ذاكرة وحدة المعالجة المركزية و/أو NVMe المتوفرة، يمكنك تفريغ كل من [حالات المحسن](https://www.deepspeed.ai/docs/config-json/#optimizer-offloading) و[المعلمات](https://www.deepspeed.ai/docs/config-json/#parameter-offloading)، أو واحدة فقط منهما، أو لا شيء. يجب أيضًا التأكد من أن `nvme_path` يشير إلى جهاز NVMe، لأنه على الرغم من أنه لا يزال يعمل مع محرك أقراص ثابت عادي أو محرك أقراص صلبة، إلا أنه سيكون أبطأ بكثير. مع NVMe الحديثة، يمكنك توقع سرعات نقل ذروة تبلغ حوالي 3.5 جيجابايت/ثانية للقراءة وحوالي 3 جيجابايت/ثانية لعمليات الكتابة. وأخيرًا، [قم بتشغيل معيار](https://github.com/microsoft/DeepSpeed/issues/998) على إعداد التدريب الخاص بك لتحديد التكوين الأمثل لـ `aio`.

يحدد ملف تهيئة ZeRO-3/Infinity أدناه معظم قيم المعلمات إلى `auto`، ولكن يمكنك أيضًا إضافة هذه القيم يدويًا.

```yml
{
"fp16": {
"enabled": "auto",
"loss_scale": 0,
"loss_scale_window": 1000,
"initial_scale_power": 16,
"hysteresis": 2,
"min_loss_scale": 1
},

"optimizer": {
"type": "AdamW",
"params": {
"lr": "auto",
"betas": "auto",
"eps": "auto",
"weight_decay": "auto"
}
},

"scheduler": {
"type": "WarmupLR",
"params": {
"warmup_min_lr": "auto",
"warmup_max_lr": "auto",
"warmup_num_steps": "auto"
}
},

"zero_optimization": {
"stage": 3,
"offload_optimizer": {
"device": "nvme",
"nvme_path": "/local_nvme",
"pin_memory": true,

"buffer_count": 4,
"fast_init": false
},
"offload_param": {
"device": "nvme",
"nvme_path": "/local_nvme",
"pin_memory": true,
"buffer_count": 5,
"buffer_size": 1e8,
"max_in_cpu": 1e9
},
"aio": {
"block_size": 262144,
"queue_depth": 32,
"thread_count": 1,
"single_submit": false,
"overlap_events": true
},
"overlap_comm": true,
"contiguous_gradients": true,
"sub_group_size": 1e9,
"reduce_bucket_size": "auto",
"stage3_prefetch_bucket_size": "auto",
"stage3_param_persistence_threshold": "auto",
"stage3_max_live_parameters": 1e9,
"stage3_max_reuse_distance": 1e9,
"stage3_gather_16bit_weights_on_model_save": true
},

"gradient_accumulation_steps": "auto",
"gradient_clipping": "auto",
"steps_per_print": 2000,
"train_batch_size": "auto",
"train_micro_batch_size_per_gpu": "auto",
"wall_clock_breakdown": false
}
```

## ميزات DeepSpeed

هناك عدد من المعلمات المهمة التي يجب تحديدها في ملف تهيئة DeepSpeed والتي يتم وصفها بإيجاز في هذا القسم.

### تفحص التنشيط/التدرج

يتضمن تفحص التنشيط والتدرج السرعة مقابل ذاكرة GPU الإضافية التي تتيح لك التغلب على السيناريوهات التي تنفد فيها ذاكرة وحدة معالجة الرسومات (GPU) الخاصة بك أو زيادة حجم دفعتك للحصول على أداء أفضل. لتمكين هذه الميزة:

1. بالنسبة لنموذج Hugging Face، قم بتعيين `model.gradient_checkpointing_enable()` أو `--gradient_checkpointing` في [`Trainer`].
2. بالنسبة للنموذج غير Hugging Face، استخدم [واجهة برمجة تطبيقات تفحص التنشيط DeepSpeed](https://deepspeed.readthedocs.io/en/latest/activation-checkpointing.html). يمكنك أيضًا استبدال رمز النمذجة Transformers واستبدال `torch.utils.checkpoint` بواجهة برمجة تطبيقات DeepSpeed. هذا النهج أكثر مرونة لأنه يمكنك تفريغ تنشيطات الإرسال إلى ذاكرة وحدة المعالجة المركزية بدلاً من إعادة حسابها.

### المحسن والمخطط

يمكن مزج محسن DeepSpeed ومحسن Transformers ومطابقتهما طالما أنك لا تقوم بتمكين `offload_optimizer`. عندما يتم تمكين `offload_optimizer`، يمكنك استخدام محسن غير DeepSpeed (باستثناء LAMB) طالما أنه يحتوي على كل من التنفيذ CPU وGPU.

<Tip warning={true}>

يمكن تعيين معلمات المحسن والمخطط لملف التكوين من سطر الأوامر لتجنب الأخطاء التي يصعب العثور عليها. على سبيل المثال، إذا تم تعيين معدل التعلم إلى قيمة مختلفة في مكان آخر، فيمكنك تجاوزه من سطر الأوامر. وبصرف النظر عن معلمات المحسن والمخطط، ستحتاج إلى التأكد من أن حجج سطر الأوامر [`Trainer`] تتطابق مع تهيئة DeepSpeed.

</Tip>

<hfoptions id="opt-sched">

<hfoption id="optimizer">

تقدم DeepSpeed العديد من [المحسنات](https://www.deepspeed.ai/docs/config-json/#optimizer-parameters) (Adam وAdamW وOneBitAdam وLAMB)، ولكن يمكنك أيضًا استيراد محسنات أخرى من PyTorch. إذا لم تقم بتكوين المحسن في التكوين، فسيقوم [`Trainer`] تلقائيًا باختيار AdamW واستخدام القيم المقدمة أو القيم الافتراضية للمعلمات التالية من سطر الأوامر: `lr`، `adam_beta1`، `adam_beta2`، `adam_epsilon`، `weight_decay`.

يمكنك تعيين المعلمات إلى `"auto"` أو إدخال قيمك المرغوبة يدويًا.

```yaml
{
"optimizer": {
"type": "AdamW",
"params": {
"lr": "auto",
"betas": "auto",
"eps": "auto",
"weight_decay": "auto"
}
}
}
```

يمكنك أيضًا استخدام محسن غير مدعوم عن طريق إضافة ما يلي إلى تهيئة المستوى الأعلى.

```yaml
{
"zero_allow_untested_optimizer": true
}
```

بدءًا من DeepSpeed==0.8.3، إذا كنت تريد استخدام التفريغ، فستحتاج أيضًا إلى إضافة ما يلي إلى تهيئة المستوى الأعلى لأن التفريغ يعمل بشكل أفضل مع محسن Adam CPU الخاص بـ DeepSpeed.

```yaml
{
"zero_force_ds_cpu_optimizer": false
}
```

</hfoption>

<hfoption id="scheduler">

تدعم DeepSpeed مخططات LRRangeTest وOneCycle وWarmupLR وWarmupDecayLR لمعدل التعلم [schedulers](https://www.deepspeed.ai/docs/config-json/#scheduler-parameters).

يوفر Transformers وDeepSpeed نفس المخططين:

* WarmupLR هو نفسه `--lr_scheduler_type constant_with_warmup` في Transformers
* WarmupDecayLR هو نفسه `--lr_scheduler_type linear` في Transformers (هذا هو المخطط الافتراضي المستخدم في Transformers)

إذا لم تقم بتكوين المخطط في التكوين، فسيقوم [`Trainer`] تلقائيًا باختيار WarmupDecayLR واستخدام القيم المقدمة أو القيم الافتراضية للمعلمات التالية من سطر الأوامر: `warmup_min_lr`، `warmup_max_lr`، `warmup_num_steps`، `total_num_steps` (يتم حسابها تلقائيًا أثناء وقت التشغيل إذا لم يتم توفير `max_steps`).

يمكنك تعيين المعلمات إلى `"auto"` أو إدخال قيمك المرغوبة يدويًا.

```yaml
{
"scheduler": {
"type": "WarmupDecayLR",
"params": {
"total_num_steps": "auto",
"warmup_min_lr": "auto",
"warmup_max_lr": "auto",
"warmup_num_steps": "auto"
}
}
}
```

</hfoption>

</hfoptions>

### الدقة

تدعم Deepspeed الدقة fp32 وfp16 وbf16 المختلطة.

<hfoptions id="precision">

<hfoption id="fp32">

إذا لم يعمل نموذجك بشكل جيد مع الدقة المختلطة، على سبيل المثال إذا لم يتم تدريبه مسبقًا في الدقة المختلطة، فقد تواجه مشكلات في الفيض أو نقصان والتي يمكن أن تسبب فقدان وظيفة الخسارة. في هذه الحالات، يجب استخدام الدقة fp32 الكاملة عن طريق تعطيل وضع fp16 الافتراضي بشكل صريح.

```yaml
{
"fp16": {
"enabled": false
}
}
```

بالنسبة لوحدات معالجة الرسوميات Ampere وإصدار PyTorch > 1.7، فإنه يتحول تلقائيًا إلى تنسيق [tf32](https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices) الأكثر كفاءة لبعض العمليات ولكن النتائج لا تزال في fp32. يمكنك التحكم فيه من [`Trainer`] عن طريق تعيين `--tf32` لتمكينه، و`--tf32 0` أو `--no_tf32` لتعطيله.

</hfoption>

<hfoption id="fp16">

لتكوين الدقة المختلطة fp16 مثل PyTorch AMP، تقلل الدقة المختلطة fp16 من استخدام الذاكرة وتسرع سرعة التدريب. يقوم [`Trainer`] بتمكين fp16 أو تعطيله تلقائيًا بناءً على قيمة `args.fp16_backend`، ويمكنك تكوين باقي التكوين. يتم تمكين fp16 من سطر الأوامر عند تمرير الحجج التالية: `--fp16`، `--fp16_backend amp` أو `--fp16_full_eval`.

```yaml
{
"fp16": {
"enabled": "auto",
"loss_scale": 0,
"loss_scale_window": 1000,
"initial_scale_power": 16,
"hysteresis": 2,
"min_loss_scale": 1
}
}
```

للحصول على خيارات تدريب fp16 الإضافية من DeepSpeed، راجع مرجع [خيارات تدريب FP16](https://www.deepspeed.ai/docs/config-json/#fp16-training-options).

لتكوين الدقة المختلطة fp16 مثل Apex، قم بتهيئة التكوين كما هو موضح أدناه باستخدام `"auto"` أو قيمك الخاصة. يقوم [`Trainer`] تلقائيًا بتكوين `amp` بناءً على قيم `args.fp16_backend` و`args.fp16_opt_level`. يمكن أيضًا تمكينه من سطر الأوامر عند تمرير الحجج التالية: `--fp16`، `--fp16_backend apex` أو `--fp16_opt_level 01`.

```yaml
{
"amp": {
"enabled": "auto",
"opt_level": "auto"
}
}
```

</hfoption>

<hfoption id="bf16">

لاستخدام bf16، ستحتاج إلى DeepSpeed==0.6.0 على الأقل. يحتوي bf16 على نفس النطاق الديناميكي مثل fp32 ولا يتطلب توسيع نطاق الخسارة. ومع ذلك، إذا كنت تستخدم [تراكم التدرجات](#gradient-accumulation) مع bf16، يتم تراكم التدرجات في bf16، وهو ما قد لا يكون مرغوبًا فيه لأن تنسيق الدقة المنخفضة هذا يمكن أن يؤدي إلى تراكم الخسارة.

يمكن إعداد bf16 في ملف التكوين أو تمكينه من سطر الأوامر عند تمرير الحجج التالية: `--bf16` أو `--bf16_full_eval`.

```yaml
{
"bf16": {
"enabled": "auto"
}
}
```

</hfoption>

</hfoptions>

### حجم الدفعة

يمكن تهيئة حجم الدفعة تلقائيًا أو تحديده صراحةً. إذا اخترت استخدام خيار `"auto"`، فسيقوم [`Trainer`] بتعيين `train_micro_batch_size_per_gpu` إلى قيمة `args.per_device_train_batch_size` و`train_batch_size` إلى `args.world_size * args.per_device_train_batch_size * args.gradient_accumulation_steps`.

```yaml
{
"train_micro_batch_size_per_gpu": "auto",
"train_batch_size": "auto"
}
```

### تراكم التدرج

يمكن تهيئة تراكم التدرج تلقائيًا أو تحديده صراحةً. إذا اخترت استخدام خيار `"auto"`، فسيقوم [`Trainer`] بتعيينه إلى قيمة `args.gradient_accumulation_steps`.

```yaml
{
"gradient_accumulation_steps": "auto"
}
```

### قص التدرج

يمكن تهيئة قص التدرج تلقائيًا أو تحديده صراحةً. إذا اخترت استخدام خيار `"auto"`، فسيقوم [`Trainer`] بتعيينه إلى قيمة `args.max_grad_norm`.

```yaml
{
"gradient_clipping": "auto"
}
```
### نوع بيانات الاتصال
بالنسبة لعمليات الاتصال الجماعي مثل عمليات التخفيض والتجمع والتشتت، يتم استخدام نوع منفصل من البيانات.

يتم تنفيذ جميع عمليات التجميع والتشتت في نفس نوع البيانات التي توجد بها البيانات. على سبيل المثال، إذا كنت تتدرب على bf16، فسيتم أيضًا تجميع البيانات في bf16 لأن التجميع عملية غير مُفقد.

عمليات التخفيض مُفقد، على سبيل المثال عندما يتم حساب المتوسطات التدرجات عبر وحدات معالجة الرسومات (GPU) متعددة. عندما يتم تنفيذ الاتصال في fp16 أو bf16، فمن المرجح أن يكون مُفقدًا لأن إضافة أرقام متعددة في دقة منخفضة ليست دقيقة. وينطبق هذا بشكل خاص على bf16 الذي يتمتع بدقة أقل من fp16. لهذا السبب، يعد fp16 هو الافتراضي لعمليات التخفيض لأن الفقدان يكون أقل عند حساب المتوسطات التدرجات.

يمكنك اختيار نوع بيانات الاتصال عن طريق تعيين معلمة "communication_data_type" في ملف التكوين. على سبيل المثال، يؤدي اختيار fp32 إلى إضافة قدر ضئيل من النفقات العامة ولكنه يضمن أن يتم تراكم عملية التخفيض في fp32 وعند الانتهاء، يتم تحويلها إلى دقة نصفية يتم التدريب عليها.

## النشر
يمكن نشر DeepSpeed بواسطة مشغلات مختلفة مثل [torchrun](https://pytorch.org/docs/stable/elastic/run.html) أو مشغل DeepSpeed أو [Accelerate](https://huggingface.co/docs/accelerate/basic_tutorials/launch#using-accelerate-launch). لنشره، أضف `--deepspeed ds_config.json` إلى سطر أوامر [`Trainer`]. يُنصح باستخدام أداة DeepSpeed [`add_config_arguments`](https://deepspeed.readthedocs.io/en/latest/initialize.html#argument-parsing) لإضافة أي حجج سطر أوامر ضرورية إلى رمزك.

سيوضح هذا الدليل كيفية نشر DeepSpeed باستخدام مشغل DeepSpeed لإعدادات التدريب المختلفة. يمكنك الاطلاع على هذا [المنشور](https://github.com/huggingface/transformers/issues/8771#issuecomment-759248400) للحصول على أمثلة أكثر عملية على الاستخدام.

<hfoptions id="deploy">
<hfoption id="multi-GPU">
لنشر DeepSpeed على وحدات معالجة الرسومات (GPU) متعددة، أضف معلمة `--num_gpus`. إذا كنت تريد استخدام جميع وحدات معالجة الرسومات (GPU) المتوفرة، فلا يلزم إضافة `--num_gpus`. يستخدم المثال أدناه وحدتي معالجة رسومات (GPU).

```bash
deepspeed --num_gpus=2 examples/pytorch/translation/run_translation.py \
--deepspeed tests/deepspeed/ds_config_zero3.json \
--model_name_or_path google-t5/t5-small --per_device_train_batch_size 1 \
--output_dir output_dir --overwrite_output_dir --fp16 \
--do_train --max_train_samples 500 --num_train_epochs 1 \
--dataset_name wmt16 --dataset_config "ro-en" \
--source_lang en --target_lang ro
```

</hfoption>
<hfoption id="single-GPU">
لنشر DeepSpeed على وحدة معالجة رسومات (GPU) واحدة، أضف معلمة `--num_gpus`. ليس من الضروري تعيين هذه القيمة صراحةً إذا كان لديك وحدة معالجة رسومات (GPU) واحدة فقط لأن DeepSpeed ينشر جميع وحدات معالجة الرسومات (GPU) التي يمكنه رؤيتها على عقدة معينة.

```bash
deepspeed --num_gpus=1 examples/pytorch/translation/run_translation.py \
--deepspeed tests/deepspeed/ds_config_zero2.json \
--model_name_or_path google-t5/t5-small --per_device_train_batch_size 1 \
--output_dir output_dir --overwrite_output_dir --fp16 \
--do_train --max_train_samples 500 --num_train_epochs 1 \
--dataset_name wmt16 --dataset_config "ro-en" \
--source_lang en --target_lang ro
```

DeepSpeed لا يزال مفيدًا مع وحدة معالجة رسومات (GPU) واحدة فقط لأنه يمكنك:

1. نقل بعض الحسابات والذاكرة إلى وحدة المعالجة المركزية (CPU) لتحرير المزيد من موارد وحدة معالجة الرسومات (GPU) لكي يستخدمها نموذجك لزيادة حجم الدُفعة أو لتناسب نموذجًا كبيرًا جدًا لا يمكن أن يناسب وحدة معالجة الرسومات (GPU) عادةً.
2. تقليل تجزئة الذاكرة باستخدام نظام إدارة ذاكرة وحدة معالجة الرسومات (GPU) الذكي الذي يسمح لك أيضًا بتناسب النماذج والدفعات الأكبر حجمًا.

<Tip>
قم بتعيين قيم `allgather_bucket_size` و`reduce_bucket_size` على 2e8 في ملف تكوين [ZeRO-2](#zero-configuration) لتحقيق أداء أفضل على وحدة معالجة رسومات (GPU) واحدة.
</Tip>

</hfoption>
</hfoptions>

### النشر على عدة عقد
العقدة هي وحدة معالجة رسومات (GPU) واحدة أو أكثر لتشغيل حمل العمل. يعد الإعداد متعدد العقد أكثر قوة ويمكن إطلاقه باستخدام مشغل DeepSpeed. بالنسبة لهذا الدليل، دعونا نفترض وجود عقدتين بثماني وحدات معالجة رسومات (GPU) لكل منهما. يمكن الوصول إلى العقدة الأولى عن طريق `ssh hostname1` والعقدة الثانية عن طريق `ssh hostname2`. يجب أن تتمكن كلتا العقدتين من التواصل مع بعضهما البعض محليًا عبر ssh بدون كلمة مرور.

افتراضيًا، يتوقع DeepSpeed أن يستخدم إعدادك متعدد العقد تخزينًا مشتركًا. إذا لم يكن الأمر كذلك ولا يمكن لكل عقدة سوى رؤية نظام الملفات المحلي، فيجب عليك ضبط ملف التكوين لإدراج [`checkpoint`](https://www.deepspeed.ai/docs/config-json/#checkpoint-options) للسماح بالتحميل بدون الوصول إلى نظام ملفات مشترك:

```yaml
{
"checkpoint": {
"use_node_local_storage": true
}
}
```

يمكنك أيضًا استخدام حجة `--save_on_each_node` الخاصة بـ [`Trainer`] لإضافة `checkpoint` أعلاه تلقائيًا إلى تكوينك.

<hfoptions id="multinode">
<hfoption id="torchrun">
بالنسبة لـ [torchrun](https://pytorch.org/docs/stable/elastic/run.html)، يجب عليك الاتصال بـ ssh بكل عقدة وتشغيل الأمر التالي على كل منهما. ينتظر المشغل حتى تتم مزامنة كلتا العقدتين قبل بدء التدريب.

```bash
torchrun --nproc_per_node=8 --nnode=2 --node_rank=0 --master_addr=hostname1 \
--master_port=9901 your_program.py <normal cl args> --deepspeed ds_config.json
```

</hfoption>
<hfoption id="deepspeed">
بالنسبة لمشغل DeepSpeed، ابدأ بإنشاء ملف `hostfile`.

```bash
hostname1 slots=8
hostname2 slots=8
```

بعد ذلك، يمكنك إطلاق التدريب باستخدام الأمر التالي. يقوم مشغل DeepSpeed تلقائيًا بتشغيل الأمر على كلتا العقدتين في نفس الوقت.

```bash
deepspeed --num_gpus 8 --num_nodes 2 --hostfile hostfile --master_addr hostname1 --master_port=9901 \
your_program.py <normal cl args> --deepspeed ds_config.json
```

راجع دليل [تكوين الموارد (متعدد العقد)](https://www.deepspeed.ai/getting-started/#resource-configuration-multi-node) لمزيد من التفاصيل حول تكوين موارد الحوسبة متعددة العقد.

</hfoption>
</hfoptions>

### SLURM
في بيئة SLURM، سيتعين عليك تكييف نص SLURM مع بيئة SLURM الخاصة بك. قد يبدو مثال نص SLURM كما يلي:

```bash
#SBATCH --job-name=test-nodes        # الاسم
#SBATCH --nodes=2                    # العقد
#SBATCH --ntasks-per-node=1          # حاسم - مهمة واحدة فقط لكل برنامج توزيع لكل عقدة!
#SBATCH --cpus-per-task=10           # عدد الأنوية لكل المهام
#SBATCH --gres=gpu:8                 # عدد وحدات معالجة الرسومات (GPU)
#SBATCH --time 20:00:00              # وقت التنفيذ الأقصى (ساعة:دقيقة:ثانية)
#SBATCH --output=%x-%j.out           # اسم ملف الإخراج

export GPUS_PER_NODE=8
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=9901

srun --jobid $SLURM_JOBID bash -c 'python -m torch.distributed.run \
--nproc_per_node $GPUS_PER_NODE --nnodes $SLURM_NNODES --node_rank $SLURM_PROCID \
--master_addr $MASTER_ADDR --master_port $MASTER_PORT \
your_program.py <normal cl args> --deepspeed ds_config.json'
```

بعد ذلك، يمكنك جدولة نشرك متعدد العقد باستخدام الأمر التالي الذي يطلق التدريب في وقت واحد على جميع العقد.

```bash
sbatch launch.slurm
```

### الدفتر
لا يدعم مشغل DeepSpeed النشر من دفتر ملاحظات، لذلك سيتعين عليك محاكاة بيئة موزعة. ومع ذلك، فإن هذا يعمل فقط لوحدة معالجة رسومات (GPU) واحدة. إذا كنت تريد استخدام أكثر من وحدة معالجة رسومات (GPU) واحدة، فيجب عليك استخدام بيئة متعددة العمليات لكي يعمل DeepSpeed. وهذا يعني أنه يتعين عليك استخدام مشغل DeepSpeed الذي لا يمكن محاكاته كما هو موضح هنا.

```py
# يتطلب DeepSpeed بيئة موزعة حتى عند استخدام عملية واحدة فقط.
# هذا يحاكي مشغلًا في الدفتر
import os

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "9994"  # غيّره إذا حدث خطأ "RuntimeError: Address already in use"
os.environ["RANK"] = "0"
os.environ["LOCAL_RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"

# تابع الآن كالمعتاد، بالإضافة إلى تمرير ملف تكوين DeepSpeed
training_args = TrainingArguments(..., deepspeed="ds_config_zero3.json")
trainer = Trainer(...)
trainer.train()
```

إذا كنت تريد إنشاء ملف التكوين أثناء التنقل في الدفتر في الدليل الحالي، فيمكنك تخصيص خلية لذلك.

```py
%%bash
cat <<'EOT' > ds_config_zero3.json
{
"fp16": {
"enabled": "auto",
"loss_scale": 0,
"loss_scale_window": 1000,
"initial_scale_power": 16,
"hysteresis": 2,
"min_loss_scale": 1
},

"optimizer": {
"type": "AdamW",
"params": {
"lr": "auto",
"betas": "auto",
"eps": "auto",
"weight_decay": "auto"
}
},

"scheduler": {
"type": "WarmupLR",
"params": {
"warmup_min_lr": "auto",
"warmup_max_lr": "auto",
"warmup_num_steps": "auto"
}
},

"zero_optimization": {
"stage": 3,
"offload_optimizer": {
"device": "cpu",
"pin_memory": true
},
"offload_param": {
"device": "cpu",
"pin_memory": true
},
"overlap_comm": true,
"contiguous_gradients": true,
"sub_group_size": 1e9,
"reduce_bucket_size": "auto",
"stage3_prefetch_bucket_size": "auto",
"stage3_param_persistence_threshold": "auto",
"stage3_max_live_parameters": 1e9,
"stage3_max_reuse_distance": 1e9,
"stage3_gather_16bit_weights_on_model_save": true
},

"gradient_accumulation_steps": "auto",
"gradient_clipping": "auto",
"steps_per_print": 2000,
"train_batch_size": "auto",
"train_micro_batch_size_per_gpu": "auto",
"wall_clock_breakdown": false
}
EOT
```

إذا كان نص التدريب موجودًا في ملف وليس في خلية دفتر ملاحظات، فيمكنك إطلاق `deepspeed` بشكل طبيعي من غلاف الدفتر في خلية دفتر ملاحظات. على سبيل المثال، لإطلاق `run_translation.py`:

```py
!git clone https://github.com/huggingface/transformers
!cd transformers; deepspeed examples/pytorch/translation/run_translation.py ...
```

يمكنك أيضًا استخدام سحر `%%bash` وكتابة رمز متعدد الأسطر لتشغيل برنامج shell، ولكن لن تتمكن من عرض السجلات حتى اكتمال التدريب. باستخدام `%%bash` السحر، لا تحتاج إلى محاكاة بيئة موزعة.

```py
%%bash

git clone https://github.com/huggingface/transformers
cd transformers
deepspeed examples/pytorch/translation/run_translation.py ...
```
بالتأكيد، سأتبع تعليماتك لترجمة النص الموجود في الفقرات والعناوين فقط:

## حفظ أوزان النموذج

يحفظ DeepSpeed أوزان الدقة الكاملة الرئيسية fp32 في ملفات نقطة تفتيش محددة مخصصة (يبدو نمط glob مثل "global_step*/*optim_states.pt") ويتم حفظها في نقطة التفتيش العادية.

### FP16

يحفظ النموذج الذي تم تدريبه باستخدام ZeRO-2 أوزان pytorch_model.bin في تنسيق fp16. ولحفظ أوزان النموذج في تنسيق fp16 لنموذج تم تدريبه باستخدام ZeRO-3، يجب عليك تعيين "stage3_gather_16bit_weights_on_model_save": true`` لأن أوزان النموذج مجزأة عبر وحدات معالجة الرسوميات (GPU) متعددة. وإلا، فإن [Trainer] لن يحفظ الأوزان في تنسيق fp16 ولن يقوم بإنشاء ملف pytorch_model.bin. ويرجع ذلك إلى أن حالة DeepSpeed تحتوي على عنصر نائب بدلاً من الأوزان الحقيقية ولن تتمكن من تحميلها.

### FP32

لا ينبغي حفظ الأوزان الدقيقة أثناء التدريب لأنها قد تتطلب الكثير من الذاكرة. وعادة ما يكون من الأفضل حفظ أوزان fp32 دون اتصال بعد اكتمال التدريب. ولكن إذا كان لديك الكثير من ذاكرة CPU الحرة، فمن الممكن حفظ أوزان fp32 أثناء التدريب. ويغطي هذا القسم كلا النهجين عبر الإنترنت وغير المتصل.

#### عبر الإنترنت

يجب أن يكون لديك حفظ نقطة تفتيش واحدة على الأقل لتحميل أحدث نقطة تفتيش كما هو موضح فيما يلي:

إذا قمت بتمكين معلمة --load_best_model_at_end لتتبع أفضل نقطة تفتيش في [TrainingArguments]، فيمكنك إنهاء التدريب أولاً وحفظ النموذج النهائي بشكل صريح. بعد ذلك، يمكنك إعادة تحميله كما هو موضح أدناه:

يمكنك أيضًا استخراج وحمّل حالة الأوزان fp32:

#### غير متصل

يوفر DeepSpeed نص برمجي zero_to_fp32.py في المستوى الأعلى من مجلد نقطة التفتيش لاستخراج الأوزان في أي نقطة. وهذا نص برمجي مستقل ولا تحتاج إلى ملف تكوين أو [Trainer].

على سبيل المثال، إذا كان مجلد نقطة التفتيش لديك يبدو كما يلي:

لإعادة بناء أوزان fp32 من نقطة تفتيش DeepSpeed (ZeRO-2 أو ZeRO-3) في المجلد الفرعي "global_step1"، قم بتشغيل الأمر التالي لإنشاء وتوحيد أوزان fp32 الكاملة من وحدات معالجة الرسوميات (GPU) متعددة في ملف pytorch_model.bin واحد. ويكتشف النص البرمجي تلقائيًا المجلد الفرعي الذي يحتوي على نقطة التفتيش.

## استدلال Zero

يضع [استدلال Zero] أوزان النموذج في ذاكرة CPU أو NVMe لتجنب إثقال كاهل وحدة معالجة الرسوميات (GPU) مما يجعل من الممكن تشغيل الاستدلال باستخدام نماذج ضخمة على وحدة معالجة الرسوميات (GPU). ولا يتطلب الاستدلال أي كميات كبيرة إضافية من الذاكرة لحالات المحسنات والتدرجات، لذلك يمكنك ملاءمة دفعات و/أو أطوال تسلسلات أكبر على نفس الأجهزة.

ويشترك استدلال Zero في نفس ملف التكوين مثل [Zero-3]، ولا تعمل تكوينات Zero-2 وZero-1 لأنها لا توفر أي فوائد للاستدلال.

لتشغيل استدلال Zero، مرر الحجج التدريبية المعتادة إلى فئة [TrainingArguments] وأضف الحجة --do_eval.

## تكامل DeepSpeed غير التابع لـ Trainer

يعمل DeepSpeed أيضًا مع Transformers دون فئة [Trainer]. ويتولى هذا التكامل [`HfDeepSpeedConfig`] الذي يعتني فقط بجمع معلمات Zero-3 وتقسيم النموذج عبر وحدات معالجة الرسوميات (GPU) متعددة عند استدعاء [`~PreTrainedModel.from_pretrained`].

لحشد Zero-3 بكفاءة، يجب إنشاء مثيل كائن [`HfDeepSpeedConfig`] قبل النموذج والاحتفاظ بذلك الكائن:

### النموذج المُدرب مسبقًا

لا يلزم وجود [`HfDeepSpeedConfig`] لـ Zero-1 أو Zero-2.

### النموذج غير المُدرب مسبقًا
### الاستدلال باستخدام ZeRO دون مدرب

لتشغيل الاستدلال باستخدام ZeRO دون استخدام المدرب في الحالات التي لا يمكنك فيها وضع النموذج على وحدة GPU واحدة، جرّب استخدام وحدات GPU إضافية و/أو تفريغ الذاكرة إلى ذاكرة الوصول العشوائي CPU. والتفصيلة المهمة التي يجب فهمها هنا هي أنه يمكنك، بفضل تصميم ZeRO، معالجة مدخلات مختلفة على وحدات GPU مختلفة بالتوازي.

تأكد من:

- تعطيل التفريغ إلى CPU إذا كانت لديك ذاكرة GPU كافية (لأنه يبطئ الأمور).
- تمكين bf16 إذا كان لديك GPU من نوع Ampere أو أحدث لتسريع الأمور. إذا لم يكن لديك أحد هذه الوحدات، فيمكنك تمكين fp16 طالما أنك لا تستخدم نموذجًا مُدربًا مسبقًا في bf16 (نماذج T5) لأنه قد يؤدي إلى خطأ في فيضان.

الق نظرة على النص البرمجي التالي للحصول على فكرة أفضل حول كيفية تشغيل الاستدلال باستخدام ZeRO دون استخدام المدرب على نموذج لا يناسب وحدة GPU واحدة.

### Generate

يتطلب استخدام وحدات GPU متعددة مع ZeRO-3 للتنمية المزامنة بين وحدات GPU عن طريق تعيين synced_gpus=True في طريقة generate في GenerationMixin. وإلا، إذا انتهت إحدى وحدات GPU من التوليد قبل الأخرى، فإن النظام بأكمله يتوقف لأن وحدات GPU المتبقية لم تتلق قطعة الوزن من وحدة GPU التي انتهت أولاً.

بالنسبة لـ Transformers>=4.28، إذا تم تعيين synced_gpus تلقائيًا على True إذا تم اكتشاف وحدات GPU متعددة أثناء التوليد.

## استكشاف الأخطاء وإصلاحها

عند مواجهة مشكلة، يجب عليك النظر فيما إذا كان DeepSpeed هو سبب المشكلة لأنه في كثير من الأحيان لا يكون كذلك (إلا إذا كان واضحًا جدًا ويمكنك رؤية وحدات DeepSpeed في الاستثناء)! يجب أن تكون الخطوة الأولى هي إعادة تشغيل إعدادك دون DeepSpeed، وإذا استمرت المشكلة، فيمكنك عندئذٍ الإبلاغ عن المشكلة. إذا كانت المشكلة متعلقة بـ DeepSpeed بشكل أساسي وغير مرتبطة بتكامل Transformers، فقم بفتح مشكلة على مستودع DeepSpeed.

بالنسبة للمشكلات المتعلقة بتكامل Transformers، يرجى تقديم المعلومات التالية:

- ملف تكوين DeepSpeed الكامل.
- وسيطات سطر الأوامر للمدرب، أو وسيطات TrainingArguments إذا كنت تقوم بإعداد المدرب بنفسك (لا تقم بإلقاء TrainingArguments الذي يحتوي على عشرات الإدخالات غير ذات الصلة).
- مخرجات ما يلي:

عندما تتعرض لمشكلة، يجب عليك النظر فيما إذا كان DeepSpeed هو سبب المشكلة لأنه في كثير من الأحيان لا يكون كذلك (إلا إذا كان واضحًا جدًا ويمكنك رؤية وحدات DeepSpeed في الاستثناء)! يجب أن تكون الخطوة الأولى هي إعادة تشغيل إعدادك دون DeepSpeed، وإذا استمرت المشكلة، فيمكنك عندئذٍ الإبلاغ عن المشكلة. إذا كانت المشكلة متعلقة بـ DeepSpeed بشكل أساسي وغير مرتبطة بتكامل Transformers، فقم بفتح مشكلة على مستودع DeepSpeed.

بالنسبة للمشكلات المتعلقة بتكامل Transformers، يرجى تقديم المعلومات التالية:

- ملف تكوين DeepSpeed الكامل.
- وسيطات سطر الأوامر للمدرب، أو وسيطات TrainingArguments إذا كنت تقوم بإعداد المدرب بنفسك (لا تقم بإلقاء TrainingArguments الذي يحتوي على عشرات الإدخالات غير ذات الصلة).
- مخرجات ما يلي:
### فقدان NaN

غالباً ما يحدث فقدان NaN عندما يكون النموذج مُدربًا مسبقًا في bf16 ثم تحاول استخدامه مع fp16 (وهو أمر ذو صلة بشكل خاص بالنماذج المدربة على TPU). لحل هذه المشكلة، استخدم fp32 أو bf16 إذا كان عتادك يدعم ذلك (TPU أو معالجات Ampere الرسومية أو الأحدث).

قد تكون المشكلة الأخرى المتعلقة باستخدام fp16. على سبيل المثال، إذا كان هذا هو تكوين fp16 الخاص بك:

قد تشاهد رسائل "OVERFLOW!" التالية في السجلات:

هذا يعني أن أداة ضبط مقياس الخسارة DeepSpeed غير قادرة على العثور على معامل قياس للتغلب على فيض الخسارة. لإصلاحه، جرب قيمة أعلى لـ "initial_scale_power" (عادةً ما تعمل 32).

## الموارد

DeepSpeed ZeRO هي تقنية قوية لتدريب وتحميل النماذج الكبيرة جدًا للتنفيذ باستخدام موارد GPU المحدودة، مما يجعلها أكثر سهولة في الوصول إلى الجميع. لمزيد من المعلومات حول DeepSpeed، يمكنك قراءة منشورات المدونة والوثائق ومستودع GitHub.

كما أن الأوراق التالية هي أيضًا مصدر رائع لمعرفة المزيد حول ZeRO:

- ZeRO: Memory Optimizations Toward Training Trillion Parameter Models
- ZeRO-Offload: Democratizing Billion-Scale Model Training
- ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning