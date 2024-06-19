# الحشو والتقطيع

غالباً ما تكون المدخلات المجمعة ذات أطوال مختلفة، لذلك لا يمكن تحويلها إلى تنسورات ذات حجم ثابت. الحشو والتشذيب هما استراتيجيتان للتعامل مع هذه المشكلة، لإنشاء تنسورات مستطيلة من دفعات ذات أطوال مختلفة. ويضيف الحشو رمز **حشو** خاص للتأكد من أن التسلسلات الأقصر ستكون بنفس الطول إما لأطول تسلسل في الدفعة أو الحد الأقصى للطول الذي يقبله النموذج. ويعمل التشذيب في الاتجاه الآخر عن طريق تشذيب التسلسلات الطويلة.

في معظم الحالات، يعمل حشو دفعتك إلى طول أطول تسلسل وتشذيبها إلى الحد الأقصى للطول الذي يمكن أن يقبله النموذج بشكل جيد إلى حد ما. ومع ذلك، تدعم واجهة برمجة التطبيقات المزيد من الاستراتيجيات إذا كنت بحاجة إليها. هناك ثلاثة حجج تحتاجها: `padding`، و`truncation`، و`max_length`.

تحكم حجة `padding` الحشو. يمكن أن يكون منطقيًا أو سلسلة:

- `True` أو `'longest'`: الحشو إلى أطول تسلسل في الدفعة (لا يتم تطبيق أي حشو إذا قمت بتوفير تسلسل واحد فقط).
- `'max_length'`: الحشو إلى طول محدد بواسطة حجة `max_length` أو الحد الأقصى للطول الذي يقبله النموذج إذا لم يتم توفير `max_length` (`max_length=None`). سيظل الحشو مطبقًا إذا قمت بتوفير تسلسل واحد فقط.
- `False` أو `'do_not_pad'`: لا يتم تطبيق أي حشو. هذا هو السلوك الافتراضي.

تحكم حجة `truncation` التشذيب. يمكن أن يكون منطقيًا أو سلسلة:

- `True` أو `'longest_first'`: التشذيب إلى طول أقصى محدد بواسطة حجة `max_length` أو الحد الأقصى للطول الذي يقبله النموذج إذا لم يتم توفير `max_length` (`max_length=None`). سيقوم هذا بتشذيب الرمز المميز حسب الرمز المميز، وإزالة رمز مميز من أطول تسلسل في الزوج حتى يتم الوصول إلى الطول الصحيح.
- `'only_second'`: التشذيب إلى طول أقصى محدد بواسطة حجة `max_length` أو الحد الأقصى للطول الذي يقبله النموذج إذا لم يتم توفير `max_length` (`max_length=None`). سيقوم هذا فقط بتشذيب الجملة الثانية من زوج إذا تم توفير زوج من التسلسلات (أو دفعة من أزواج التسلسلات).
- `'only_first'`: التشذيب إلى طول أقصى محدد بواسطة حجة `max_length` أو الحد الأقصى للطول الذي يقبله النموذج إذا لم يتم توفير `max_length` (`max_length=None`). سيقوم هذا فقط بتشذيب الجملة الأولى من زوج إذا تم توفير زوج من التسلسلات (أو دفعة من أزواج التسلسلات).
- `False` أو `'do_not_truncate'`: لا يتم تطبيق أي تشذيب. هذا هو السلوك الافتراضي.

تحكم حجة `max_length` طول الحشو والتشذيب. يمكن أن يكون عدد صحيح أو `None`، وفي هذه الحالة سيتم تعيينه افتراضيًا إلى الحد الأقصى للطول الذي يمكن أن يقبله النموذج. إذا لم يكن للنموذج طول إدخال محدد، يتم إلغاء تنشيط التشذيب أو الحشو إلى `max_length`.

يوضح الجدول التالي الطريقة الموصى بها لإعداد الحشو والتشذيب. إذا كنت تستخدم أزواج تسلسلات الإدخال في أي من الأمثلة التالية، فيمكنك استبدال `truncation=True` بـ `STRATEGY` المحددة في `['only_first'، 'only_second'، 'longest_first']`، أي `truncation='only_second'` أو `truncation='longest_first'` للتحكم في كيفية تشذيب كل من التسلسلات في الزوج كما هو مفصل سابقًا.

| التشذيب | الحشو | التعليمات |
| --------- | ------ | ---------- |
| لا تشذيب | لا حشو | `tokenizer(batch_sentences)` |
| | الحشو إلى أقصى طول تسلسل في الدفعة | `tokenizer(batch_sentences، padding=True)` أو |
| | | `tokenizer(batch_sentences، padding='longest')` |
| | الحشو إلى أقصى طول إدخال نموذج | `tokenizer(batch_sentences، padding='max_length')` |
| | الحشو إلى طول محدد | `tokenizer(batch_sentences، padding='max_length'، max_length=42)` |
| | الحشو إلى مضاعف لقيمة | `tokenizer(batch_sentences، padding=True، pad_to_multiple_of=8)` |
| التشذيب إلى أقصى طول إدخال نموذج | لا حشو | `tokenizer(batch_sentences، truncation=True)` أو |
| | | `tokenizer(batch_sentences، truncation=STRATEGY)` |
| | الحشو إلى أقصى طول تسلسل في الدفعة | `tokenizer(batch_sentences، padding=True، truncation=True)` أو |
| | | `tokenizer(batch_sentences، padding=True، truncation=STRATEGY)` |
| | الحشو إلى أقصى طول إدخال نموذج | `tokenizer(batch_sentences، padding='max_length'، truncation=True)` أو |
| | | `tokenizer(batch_sentences، padding='max_length'، truncation=STRATEGY)` |
| | الحشو إلى طول محدد | غير ممكن |
| التشذيب إلى طول محدد | لا حشو | `tokenizer(batch_sentences، truncation=True، max_length=42)` أو |
| | | `tokenizer(batch_sentences، truncation=STRATEGY، max_length=42)` |
| | الحشو إلى أقصى طول تسلسل في الدفعة | `tokenizer(batch_sentences، padding=True، truncation=True، max_length=42)` أو |
| | | `tokenizer(batch_sentences، padding=True، truncation=STRATEGY، max_length=42)` |
| | الحشو إلى أقصى طول إدخال نموذج | غير ممكن |
| | الحشو إلى طول محدد | `tokenizer(batch_sentences، padding='max_length'، truncation=True، max_length=42)` أو |
| | | `tokenizer(batch_sentences، padding='max_length'، truncation=STRATEGY، max_length=42)` |