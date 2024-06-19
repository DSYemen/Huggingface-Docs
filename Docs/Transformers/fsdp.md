# Fully Sharded Data Parallel

تعد طريقة Fully Sharded Data Parallel (FSDP) طريقة موازية للبيانات تقوم بتجزئة معلمات النموذج والتدرجات وحالات المحسن عبر عدد وحدات معالجة الرسوميات (GPU) المتوفرة (يطلق عليها أيضًا اسم workers أو *rank*). على عكس DistributedDataParallel (DDP)، يقلل FSDP من استخدام الذاكرة لأن النموذج يتم تكراره على كل GPU. وهذا يحسن كفاءة ذاكرة GPU ويسمح بتدريب نماذج أكبر بكثير على عدد أقل من وحدات معالجة الرسوميات. تم دمج FSDP مع Accelerate، وهي مكتبة لإدارة التدريب بسهولة في بيئات موزعة، مما يعني أنه متاح للاستخدام من فئة ['Trainer'].

قبل البدء، تأكد من تثبيت Accelerate وPyTorch 2.1.0 أو أحدث.

```bash
pip install accelerate
```

## تكوين FSDP

لبدء الاستخدام، قم بتشغيل أمر [`accelerate config`](https://huggingface.co/docs/accelerate/package_reference/cli#accelerate-config) لإنشاء ملف تكوين لبيئة التدريب الخاصة بك. يستخدم Accelerate ملف التكوين هذا لإعداد بيئة التدريب الصحيحة تلقائيًا بناءً على خيارات التدريب التي حددتها في `accelerate config`.

```bash
accelerate config
```

عند تشغيل `accelerate config`، ستتم مطالبتك بسلسلة من الخيارات لتكوين بيئة التدريب الخاصة بك. يغطي هذا القسم بعض أهم خيارات FSDP. لمزيد من المعلومات حول خيارات FSDP الأخرى المتاحة، راجع معلمات [fsdp_config](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments.fsdp_config).

### استراتيجية التجزئة

يقدم FSDP عددًا من استراتيجيات التجزئة للاختيار من بينها:

- `FULL_SHARD` - تجزئة معلمات النموذج والتدرجات وحالات المحسن عبر workers؛ حدد `1` لهذا الخيار
- `SHARD_GRAD_OP` - تجزئة التدرجات وحالات المحسن عبر workers؛ حدد `2` لهذا الخيار
- `NO_SHARD` - لا تقم بالتجزئة (وهذا ما يعادل DDP)؛ حدد `3` لهذا الخيار
- `HYBRID_SHARD` - تجزئة معلمات النموذج والتدرجات وحالات المحسن داخل كل worker حيث يحتوي كل worker أيضًا على نسخة كاملة؛ حدد `4` لهذا الخيار
- `HYBRID_SHARD_ZERO2` - تجزئة التدرجات وحالات المحسن داخل كل worker حيث يحتوي كل worker أيضًا على نسخة كاملة؛ حدد `5` لهذا الخيار

يتم تمكين هذه الميزة عن طريق علم `fsdp_sharding_strategy`.

### تفريغ الذاكرة المؤقتة إلى وحدة المعالجة المركزية (CPU)

يمكنك أيضًا تفريغ المعلمات والتدرجات إلى وحدة المعالجة المركزية (CPU) عندما لا تكون قيد الاستخدام لتوفير المزيد من ذاكرة GPU والمساعدة في تثبيت النماذج الكبيرة حيث قد لا يكون FSDP كافيًا. يتم تمكين هذه الميزة عن طريق تعيين `fsdp_offload_params: true` عند تشغيل `accelerate config`.

### سياسة التغليف

يتم تطبيق FSDP عن طريق تغليف كل طبقة في الشبكة. يتم تطبيق التغليف عادةً بطريقة متداخلة حيث يتم التخلص من الأوزان الكاملة بعد كل تمرير للأمام لتوفير الذاكرة للاستخدام في الطبقة التالية. تعتبر سياسة التغليف التلقائي أبسط طريقة لتنفيذ ذلك ولا تحتاج إلى تغيير أي كود. يجب عليك تحديد `fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP` لتغليف طبقة المحول و`fsdp_transformer_layer_cls_to_wrap` لتحديد الطبقة التي سيتم تغليفها (على سبيل المثال `BertLayer`).

من ناحية أخرى، يمكنك اختيار سياسة التغليف المستندة إلى الحجم حيث يتم تطبيق FSDP على طبقة إذا تجاوزت عددًا معينًا من المعلمات. يتم تمكين هذه الميزة عن طريق تعيين `fsdp_wrap_policy: SIZE_BASED_WRAP` و`min_num_param` إلى عتبة الحجم المرغوبة.

### نقاط التفتيش

يجب حفظ نقاط التفتيش الوسيطة باستخدام `fsdp_state_dict_type: SHARDED_STATE_DICT` لأن حفظ القاموس الكامل لحالة النموذج مع تفريغ الذاكرة المؤقتة إلى وحدة المعالجة المركزية (CPU) على الرتبة 0 يستغرق وقتًا طويلاً وغالبًا ما يؤدي إلى أخطاء "NCCL Timeout" بسبب التعليق غير المحدد أثناء البث. يمكنك استئناف التدريب باستخدام قواميس الحالة المجزأة مع طريقة [`~accelerate.Accelerator.load_state`].

```py
# الدليل الذي يحتوي على نقاط التفتيش
accelerator.load_state("ckpt")
```

ومع ذلك، عندما ينتهي التدريب، تريد حفظ قاموس الحالة الكامل لأن قاموس الحالة المجزأ متوافق فقط مع FSDP.

```بي
إذا trainer.is_fsdp_enabled:
trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")

trainer.save_model(script_args.output_dir)
```

### وحدة معالجة الرسومات القابلة للبرمجة (TPU)

تدعم PyTorch XLA تدريب FSDP لـ TPUs ويمكن تمكينه عن طريق تعديل ملف تكوين FSDP الذي تم إنشاؤه بواسطة `accelerate config`. بالإضافة إلى استراتيجيات التجزئة وخيارات التغليف المحددة أعلاه، يمكنك إضافة المعلمات الموضحة أدناه إلى الملف.

```yaml
xla: True # يجب تعيينها على True لتمكين PyTorch/XLA
xla_fsdp_settings: # إعدادات FSDP المحددة لـ XLA
xla_fsdp_grad_ckpt: True # استخدام نقطة تفتيش التدرج
```

تسمح لك [`xla_fsdp_settings`](https://github.com/pytorch/xla/blob/2e6e183e0724818f137c8135b34ef273dea33318/torch_xla/distributed/fsdp/xla_fully_sharded_data_parallel.py#L128) بتكوين معلمات إضافية محددة لـ XLA لـ FSDP.

## بدء التدريب

قد يبدو ملف تكوين FSDP على النحو التالي:

```yaml
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: FSDP
downcast_bf16: 'no'
fsdp_config:
fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
fsdp_backward_prefetch_policy: BACKWARD_PRE
fsd
p_cpu_ram_efficient_loading: true
fsdp_forward_prefetch: false
fsdp_offload_params: true
fsdp_sharding_strategy: 1
fsdp_state_dict_type: SHARDED_STATE_DICT
fsdp_sync_module_states: true
fsdp_transformer_layer_cls_to_wrap: BertLayer
fsdp_use_orig_params: true
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 2
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

لبدء التدريب، قم بتشغيل أمر [`accelerate launch`](https://huggingface.co/docs/accelerate/package_reference/cli#accelerate-launch) وسيستخدم تلقائيًا ملف التكوين الذي قمت بإنشائه سابقًا باستخدام `accelerate config`.

```bash
accelerate launch my-trainer-script.py
```

```bash
accelerate launch --fsdp="full shard" --fsdp_config="path/to/fsdp_config/ my-trainer-script.py
```

## الخطوات التالية

يمكن أن يكون FSDP أداة قوية لتدريب النماذج الكبيرة جدًا ولديك إمكانية الوصول إلى أكثر من وحدة معالجة رسوميات (GPU) واحدة أو وحدة معالجة الرسوميات القابلة للبرمجة (TPU). من خلال تجزئة معلمات النموذج وحالات المحسن والتدرجات، وحتى تفريغها إلى وحدة المعالجة المركزية (CPU) عندما تكون غير نشطة، يمكن أن يقلل FSDP من التكلفة العالية للتدريب على نطاق واسع. إذا كنت مهتمًا بمعرفة المزيد، فقد يكون ما يلي مفيدًا:

- اتبع الدليل الأكثر عمقًا لـ Accelerate لـ [FSDP](https://huggingface.co/docs/accelerate/usage_guides/fsdp).
- اقرأ منشور المدونة [تقديم واجهة برمجة التطبيقات PyTorch Fully Sharded Data Parallel (FSDP)](https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/).
- اقرأ منشور المدونة [توسيع نطاق نماذج PyTorch على وحدات معالجة الرسوميات القابلة للبرمجة (TPU) السحابية باستخدام FSDP](https://pytorch.org/blog/scaling-pytorch-models-on-cloud-tpus-with-fsdp/).