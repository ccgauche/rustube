use std::{fs::OpenOptions, io::Write};

use alloc::sync::Arc;
use regex::Regex;

#[derive(Clone)]
enum Value {
    String(String),
    Int(i32),
    Null,
    List(Vec<Self>),
    FunctionTwoMut(Arc<Box<dyn Fn(&mut Self, Self) -> Self>>),
}

impl core::fmt::Debug for Value {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::String(arg0) => f.debug_tuple("String").field(arg0).finish(),
            Self::Int(arg0) => f.debug_tuple("Int").field(arg0).finish(),
            Self::Null => write!(f, "Null"),
            Self::List(arg0) => f.debug_tuple("List").field(arg0).finish(),
            Self::FunctionTwoMut(_) => f.debug_tuple("FunctionTwoMut").finish(),
        }
    }
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::String(l0), Self::String(r0)) => l0 == r0,
            (Self::Int(l0), Self::Int(r0)) => l0 == r0,
            (Self::List(l0), Self::List(r0)) => l0 == r0,
            _ => false,
        }
    }
}

impl Value {
    fn exec2_mut(&self, a: &mut Self, b: Self) -> Self {
        match self {
            Value::FunctionTwoMut(f) => f(a, b),
            _ => panic!("Not a function"),
        }
    }

    fn as_str(&self) -> &str {
        match self {
            Value::String(s) => s,
            _ => panic!("Not a string"),
        }
    }
}

pub(crate) struct Cipher {
    throttling_plan: Vec<(i32, i32, Option<i32>)>,
    throttling_array: Vec<Value>,
    calculated_n: Option<String>,
}

impl Cipher {
    pub(crate) fn new(js: &str) -> Self {
        let throttling_plan = get_throttling_plan(js);
        let throttling_array = get_throttling_function_array(js);

        Cipher {
            throttling_plan,
            throttling_array,
            calculated_n: None,
        }
    }

    pub(crate) fn calculate_n(&mut self, initial_n: Vec<char>) -> String {
        if self.calculated_n.is_some() {
            return self.calculated_n.clone().unwrap();
        }

        // First, update all instances of 'b' with the list(initial_n)
        for i in 0..self.throttling_array.len() {
            if let Value::String(b) = &self.throttling_array[i] {
                if b == "b" {
                    self.throttling_array[i] = Value::List(
                        initial_n
                            .iter()
                            .map(|x| Value::String(x.to_string()))
                            .collect(),
                    );
                }
            }
        }

        for step in &self.throttling_plan {
            let curr_func: Value = self.throttling_array[step.0 as usize].clone();

            // if !curr_func.starts_with("function") {
            //     panic!("{} is not callable.", curr_func);
            // }

            if let Some(e) = &step.2 {
                let second_arg = self.throttling_array[*e as usize].clone();
                let first_arg = &mut self.throttling_array[step.1 as usize];
                curr_func.exec2_mut(first_arg, second_arg);
            } else {
                let first_arg = &mut self.throttling_array[step.1 as usize];
                curr_func.exec2_mut(first_arg, Value::Null);
            }
        }

        self.calculated_n = Some(self.throttling_array[0].as_str().to_string());
        self.calculated_n.clone().unwrap()
    }
}

/* def get_throttling_function_name(js: str) -> str:
    """Extract the name of the function that computes the throttling parameter.

    :param str js:
        The contents of the base.js asset file.
    :rtype: str
    :returns:
        The name of the function used to compute the throttling parameter.
    """
    function_patterns = [
        # https://github.com/ytdl-org/youtube-dl/issues/29326#issuecomment-865985377
        # https://github.com/yt-dlp/yt-dlp/commit/48416bc4a8f1d5ff07d5977659cb8ece7640dcd8
        # var Bpa = [iha];
        # ...
        # a.C && (b = a.get("n")) && (b = Bpa[0](b), a.set("n", b),
        # Bpa.length || iha("")) }};
        # In the above case, `iha` is the relevant function name
        r'a\.[a-zA-Z]\s*&&\s*\([a-z]\s*=\s*a\.get\("n"\)\)\s*&&\s*'
        r'\([a-z]\s*=\s*([a-zA-Z0-9$]+)(\[\d+\])?\([a-z]\)',
    ]
    logger.debug('Finding throttling function name')
    for pattern in function_patterns:
        regex = re.compile(pattern)
        function_match = regex.search(js)
        if function_match:
            logger.debug("finished regex search, matched: %s", pattern)
            if len(function_match.groups()) == 1:
                return function_match.group(1)
            idx = function_match.group(2)
            if idx:
                idx = idx.strip("[]")
                array = re.search(
                    r'var {nfunc}\s*=\s*(\[.+?\]);'.format(
                        nfunc=re.escape(function_match.group(1))),
                    js
                )
                if array:
                    array = array.group(1).strip("[]").split(",")
                    array = [x.strip() for x in array]
                    return array[int(idx)]

    raise RegexMatchError(
        caller="get_throttling_function_name", pattern="multiple"
    )
) */

fn get_throttling_function_name(js: &str) -> String {
    let function_patterns =
        [r#"\.get\("n"\)\)&&\(b=(?P<nfunc>[a-zA-Z0-9$]{3})(\[(?P<idx>\d+)\])?\([a-zA-Z0-9]\)"#];
    for pattern in function_patterns.iter() {
        let regex = Regex::new(pattern).unwrap();
        let function_match = regex.captures_iter(js);
        for capture in function_match {
            if capture.name("idx").is_some() {
                let nfunc = capture.name("nfunc").unwrap().as_str();
                std::fs::write("dbg1.txt", nfunc).unwrap();
                let array = Regex::new(&format!(r#"var {nfunc}\s*=\s*\[(.+?)\];"#,))
                    .unwrap()
                    .captures_iter(js)
                    .next()
                    .unwrap()[1]
                    .to_string();

                return array;
            } else {
                return capture.name("nfunc").unwrap().as_str().to_string();
            }
        }
    }
    panic!("get_throttling_function_name failed");
}

/*
def get_throttling_function_code(js: str) -> str:
    """Extract the raw code for the throttling function.

    :param str js:
        The contents of the base.js asset file.
    :rtype: str
    :returns:
        The name of the function used to compute the throttling parameter.
    """
    # Begin by extracting the correct function name
    name = re.escape(get_throttling_function_name(js))

    # Identify where the function is defined
    pattern_start = r"%s=function\(\w\)" % name
    regex = re.compile(pattern_start)
    match = regex.search(js)

    # Extract the code within curly braces for the function itself, and merge any split lines
    code_lines_list = find_object_from_startpoint(js, match.span()[1]).split('\n')
    joined_lines = "".join(code_lines_list)

    # Prepend function definition (e.g. `Dea=function(a)`)
    return match.group(0) + joined_lines */

fn get_throttling_function_code(js: &str) -> String {
    let name = get_throttling_function_name(js);
    let pattern_start = format!(r#"{}=function\(\w\)"#, name);
    std::fs::write("pattern.txt", pattern_start.clone()).unwrap();
    let regex = Regex::new(&pattern_start).unwrap();
    let matched = regex.find(js).unwrap();
    let joined_lines = find_object(&js[matched.end()..])
        .split('\n')
        .collect::<Vec<&str>>()
        .join("");
    format!("{}{}", matched.as_str(), joined_lines)
}

/* def get_throttling_function_array(js: str) -> List[Any]:
"""Extract the "c" array.

:param str js:
    The contents of the base.js asset file.
:returns:
    The array of various integers, arrays, and functions.
"""
raw_code = get_throttling_function_code(js)

array_start = r",c=\["
array_regex = re.compile(array_start)
match = array_regex.search(raw_code)

array_raw = find_object_from_startpoint(raw_code, match.span()[1] - 1)
str_array = throttling_array_split(array_raw)

converted_array = []
for el in str_array:
    try:
        converted_array.append(int(el))
        continue
    except ValueError:
        # Not an integer value.
        pass

    if el == 'null':
        converted_array.append(None)
        continue

    if el.startswith('"') and el.endswith('"'):
        # Convert e.g. '"abcdef"' to string without quotation marks, 'abcdef'
        converted_array.append(el[1:-1])
        continue

    if el.startswith('function'):
        mapper = (
            (r"{for\(\w=\(\w%\w\.length\+\w\.length\)%\w\.length;\w--;\)\w\.unshift\(\w.pop\(\)\)}", throttling_unshift),  # noqa:E501
            (r"{\w\.reverse\(\)}", throttling_reverse),
            (r"{\w\.push\(\w\)}", throttling_push),
            (r";var\s\w=\w\[0\];\w\[0\]=\w\[\w\];\w\[\w\]=\w}", throttling_swap),
            (r"case\s\d+", throttling_cipher_function),
            (r"\w\.splice\(0,1,\w\.splice\(\w,1,\w\[0\]\)\[0\]\)", throttling_nested_splice),  # noqa:E501
            (r";\w\.splice\(\w,1\)}", js_splice),
            (r"\w\.splice\(-\w\)\.reverse\(\)\.forEach\(function\(\w\){\w\.unshift\(\w\)}\)", throttling_prepend),  # noqa:E501
            (r"for\(var \w=\w\.length;\w;\)\w\.push\(\w\.splice\(--\w,1\)\[0\]\)}", throttling_reverse),  # noqa:E501
        )

        found = False
        for pattern, fn in mapper:
            if re.search(pattern, el):
                converted_array.append(fn)
                found = True
        if found:
            continue

    converted_array.append(el)

# Replace null elements with array itself
for i in range(len(converted_array)):
    if converted_array[i] is None:
        converted_array[i] = converted_array

return converted_array */

fn get_throttling_function_array(js: &str) -> Vec<Value> {
    let raw_code = get_throttling_function_code(js);
    let array_start = r#",c=\["#;
    let array_regex = Regex::new(array_start).unwrap();
    let matched = array_regex.find(&raw_code).unwrap();
    let array_raw = find_object(&raw_code[matched.end() - 1..]);
    let str_array = throttling_array_split(&array_raw);
    let mut converted_array = vec![];
    for el in str_array {
        if let Ok(int) = el.parse::<i32>() {
            converted_array.push(Value::Int(int));
            continue;
        }
        if el == "null" {
            converted_array.push(Value::Null);
            continue;
        }
        if el.starts_with('"') && el.ends_with('"') {
            converted_array.push(Value::String(el[1..el.len() - 1].to_string()));
            continue;
        }
        if el.starts_with("function") {
            let mapper = vec![
                (
                    r#"\{for\(\w=\(\w%\w\.length\+\w\.length\)%\w\.length;\w--;\)\w\.unshift\(\w.pop\(\)\)\}"#,
                    Value::FunctionTwoMut(Arc::new(Box::new(throttling_unshift))),
                ), // noqa:E501
                (
                    r#"\{\w\.reverse\(\)\}"#,
                    Value::FunctionTwoMut(Arc::new(Box::new(throttling_reverse))),
                ),
                (
                    r#"\{\w\.push\(\w\)\}"#,
                    Value::FunctionTwoMut(Arc::new(Box::new(throttling_push))),
                ),
                (
                    r#";var\s\w=\w\[0\];\w\[0\]=\w\[\w\];\w\[\w\]=\w}"#,
                    Value::FunctionTwoMut(Arc::new(Box::new(throttling_swap))),
                ),
                (
                    r#"case\s\d+"#,
                    Value::FunctionTwoMut(Arc::new(Box::new(throttling_cipher_function))),
                ),
                (
                    r#"\w\.splice\(0,1,\w\.splice\(\w,1,\w\[0\]\)\[0\]\)"#,
                    Value::FunctionTwoMut(Arc::new(Box::new(throttling_nested_splice))),
                ), // noqa:E501
                (
                    r#";\w\.splice\(\w,1\)}"#,
                    Value::FunctionTwoMut(Arc::new(Box::new(|a, b| {
                        js_splice(a, b, Value::Null, Value::List(vec![]))
                    }))),
                ),
                (
                    r#"\w\.splice\(-\w\)\.reverse\(\)\.forEach\(function\(\w\)\{\w\.unshift\(\w\)\}\)"#,
                    Value::FunctionTwoMut(Arc::new(Box::new(throttling_prepend))),
                ), // noqa:E501
                (
                    r#"for\(var \w=\w\.length;\w;\)\w\.push\(\w\.splice\(--\w,1\)\[0\]\)}"#,
                    Value::FunctionTwoMut(Arc::new(Box::new(throttling_reverse))),
                ), // noqa:E501
            ];

            let mut found = false;
            for (pattern, func) in mapper {
                if Regex::new(pattern).unwrap().is_match(&el) {
                    converted_array.push(func);
                    found = true;
                }
            }
            if found {
                continue;
            }
        }
        converted_array.push(Value::String(el));
    }

    // Replace null elements with array itself

    for i in 0..converted_array.len() {
        if matches!(converted_array[i], Value::Null) {
            converted_array[i] = Value::List(converted_array.clone());
        }
    }

    converted_array
}

/* def get_throttling_plan(js: str):
"""Extract the "throttling plan".

The "throttling plan" is a list of tuples used for calling functions
in the c array. The first element of the tuple is the index of the
function to call, and any remaining elements of the tuple are arguments
to pass to that function.

:param str js:
    The contents of the base.js asset file.
:returns:
    The full function code for computing the throttlign parameter.
"""
raw_code = get_throttling_function_code(js)

transform_start = r"try{"
plan_regex = re.compile(transform_start)
match = plan_regex.search(raw_code)

transform_plan_raw = find_object_from_startpoint(raw_code, match.span()[1] - 1)

# Steps are either c[x](c[y]) or c[x](c[y],c[z])
step_start = r"c\[(\d+)\]\(c\[(\d+)\](,c(\[(\d+)\]))?\)"
step_regex = re.compile(step_start)
matches = step_regex.findall(transform_plan_raw)
transform_steps = []
for match in matches:
    if match[4] != '':
        transform_steps.append((match[0],match[1],match[4]))
    else:
        transform_steps.append((match[0],match[1]))

return transform_steps */

fn get_throttling_plan(js: &str) -> Vec<(i32, i32, Option<i32>)> {
    let raw_code = get_throttling_function_code(js);
    let transform_start = r#"try\{"#;
    std::fs::write("debug.1.js", &raw_code).unwrap();
    let plan_regex = Regex::new(transform_start).unwrap();
    let matched = plan_regex.find(&raw_code).unwrap();
    let transform_plan_raw = find_object(&raw_code[matched.end() - 1..]);
    let step_start = r#"c\[(\d+)\]\(c\[(\d+)\](,c(\[(\d+)\]))?\)"#;
    let step_regex = Regex::new(step_start).unwrap();
    let mut transform_steps = vec![];
    for cap in step_regex.captures_iter(&transform_plan_raw) {
        if cap.get(4).is_some() {
            transform_steps.push((
                cap[1].parse::<i32>().unwrap(),
                cap[2].parse::<i32>().unwrap(),
                Some(cap[4].parse::<i32>().unwrap()),
            ));
        } else {
            transform_steps.push((
                cap[1].parse::<i32>().unwrap(),
                cap[2].parse::<i32>().unwrap(),
                None,
            ));
        }
    }
    transform_steps
}

/* def throttling_reverse(arr: list):
"""Reverses the input list.

Needs to do an in-place reversal so that the passed list gets changed.
To accomplish this, we create a reversed copy, and then change each
indvidual element.
"""
reverse_copy = arr.copy()[::-1]
for i in range(len(reverse_copy)):
    arr[i] = reverse_copy[i] */

fn throttling_reverse(arr: &mut Value, _: Value) -> Value {
    match arr {
        Value::List(arr) => {
            arr.reverse();
        }
        _ => panic!("throttling_reverse() expects a list"),
    }
    Value::Null
}

/* def throttling_push(d: list, e: Any):
"""Pushes an element onto a list."""
d.append(e) */

fn throttling_push(d: &mut Value, e: Value) -> Value {
    match d {
        Value::List(d) => {
            d.push(e);
        }
        _ => panic!("throttling_push() expects a list"),
    }
    Value::Null
}

/* def throttling_mod_func(d: list, e: int):
"""Perform the modular function from the throttling array functions.

In the javascript, the modular operation is as follows:
e = (e % d.length + d.length) % d.length

We simply translate this to python here.
"""
return (e % len(d) + len(d)) % len(d) */

fn throttling_mod_func(d: &[Value], e: i32) -> i32 {
    (e % d.len() as i32 + d.len() as i32) % d.len() as i32
}

/* def throttling_unshift(d: list, e: int):
"""Rotates the elements of the list to the right.

In the javascript, the operation is as follows:
for(e=(e%d.length+d.length)%d.length;e--;)d.unshift(d.pop())
"""
e = throttling_mod_func(d, e)
new_arr = d[-e:] + d[:-e]
d.clear()
for el in new_arr:
    d.append(el) */

fn throttling_unshift(d: &mut Value, e: Value) -> Value {
    match (d, e) {
        (Value::List(d), Value::Int(e)) => {
            let e = throttling_mod_func(&d, e);
            let mut new_arr = d.split_off(d.len() - e as usize);
            new_arr.append(d);
            *d = new_arr;
        }
        _ => panic!("throttling_unshift() expects a list"),
    }
    Value::Null
}

/*
def throttling_cipher_function(d: list, e: str):
    """This ciphers d with e to generate a new list.

    In the javascript, the operation is as follows:
    var h = [A-Za-z0-9-_], f = 96;  // simplified from switch-case loop
    d.forEach(
        function(l,m,n){
            this.push(
                n[m]=h[
                    (h.indexOf(l)-h.indexOf(this[m])+m-32+f--)%h.length
                ]
            )
        },
        e.split("")
    )
    """
    h = list('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_')
    f = 96
    # by naming it "this" we can more closely reflect the js
    this = list(e)

    # This is so we don't run into weirdness with enumerate while
    #  we change the input list
    copied_list = d.copy()

    for m, l in enumerate(copied_list):
        bracket_val = (h.index(l) - h.index(this[m]) + m - 32 + f) % len(h)
        this.append(
            h[bracket_val]
        )
        d[m] = h[bracket_val]
        f -= 1
    */

fn throttling_cipher_function(d: &mut Value, e: Value) -> Value {
    match (d, e) {
        (Value::List(d), Value::String(e)) => {
            let h = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
            let mut f = 96;
            let mut this = e.chars().collect::<Vec<char>>();
            let copied_list = d.clone();
            for (m, l) in copied_list.iter().enumerate() {
                if let Value::String(l) = l {
                    let bracket_val = (h.find(l).unwrap() as i32 - h.find(this[m]).unwrap() as i32
                        + m as i32
                        - 32
                        + f)
                        % h.len() as i32;
                    this.push(h.chars().nth(bracket_val as usize).unwrap());
                    d[m] = Value::String(h.chars().nth(bracket_val as usize).unwrap().to_string());
                    f -= 1;
                } else {
                    panic!("");
                }
            }
        }
        _ => panic!("throttling_cipher_function() expects a list"),
    }
    Value::Null
}

/* def throttling_nested_splice(d: list, e: int):
"""Nested splice function in throttling js.

In the javascript, the operation is as follows:
function(d,e){
    e=(e%d.length+d.length)%d.length;
    d.splice(
        0,
        1,
        d.splice(
            e,
            1,
            d[0]
        )[0]
    )
}

While testing, all this seemed to do is swap element 0 and e,
but the actual process is preserved in case there was an edge
case that was not considered.
"""
e = throttling_mod_func(d, e)
inner_splice = js_splice(
    d,
    e,
    1,
    d[0]
)
js_splice(
    d,
    0,
    1,
    inner_splice[0]
) */

fn throttling_nested_splice(d: &mut Value, e: Value) -> Value {
    match (d.clone(), e) {
        (Value::List(d1), Value::Int(e)) => {
            let e = throttling_mod_func(&d1, e);
            // TODO: check if this is correct (the last argument should be a list)
            let inner_splice = js_splice(
                d,
                Value::Int(e),
                Value::Int(1),
                Value::List(vec![d1[0].clone()]),
            );
            js_splice(
                d,
                Value::Int(0),
                Value::Int(1),
                Value::List(vec![if let Value::List(v) = inner_splice {
                    v[0].clone()
                } else {
                    panic!("throttling_nested_splice() expects a list")
                }]),
            )
        }
        _ => panic!("throttling_nested_splice() expects a list"),
    }
}

/* def throttling_prepend(d: list, e: int):
"""

In the javascript, the operation is as follows:
function(d,e){
    e=(e%d.length+d.length)%d.length;
    d.splice(-e).reverse().forEach(
        function(f){
            d.unshift(f)
        }
    )
}

Effectively, this moves the last e elements of d to the beginning.
"""
start_len = len(d)
# First, calculate e
e = throttling_mod_func(d, e)

# Then do the prepending
new_arr = d[-e:] + d[:-e]

# And update the input list
d.clear()
for el in new_arr:
    d.append(el)

end_len = len(d)
assert start_len == end_len */

fn throttling_prepend(d: &mut Value, e: Value) -> Value {
    match (d, e) {
        (Value::List(d), Value::Int(e)) => {
            let start_len = d.len();
            let e = throttling_mod_func(d, e);
            let mut new_arr = d.split_off(d.len() - e as usize);
            new_arr.append(d);
            *d = new_arr;
            let end_len = d.len();
            assert_eq!(start_len, end_len);
        }
        _ => panic!("throttling_prepend() expects a list"),
    }
    Value::Null
}

/*
def test_throttling_prepend():
    a = [1, 2, 3, 4]
    cipher.throttling_prepend(a, 1)
    assert a == [4, 1, 2, 3]
    a = [1, 2, 3, 4]
    cipher.throttling_prepend(a, 2)
    assert a == [3, 4, 1, 2]
*/
#[test]
fn test_throttling_prepend() {
    let mut a = Value::List(vec![
        Value::Int(1),
        Value::Int(2),
        Value::Int(3),
        Value::Int(4),
    ]);
    throttling_prepend(&mut a, Value::Int(1));
    assert_eq!(
        a,
        Value::List(vec![
            Value::Int(4),
            Value::Int(1),
            Value::Int(2),
            Value::Int(3)
        ])
    );
    let mut a = Value::List(vec![
        Value::Int(1),
        Value::Int(2),
        Value::Int(3),
        Value::Int(4),
    ]);
    throttling_prepend(&mut a, Value::Int(2));
    assert_eq!(
        a,
        Value::List(vec![
            Value::Int(3),
            Value::Int(4),
            Value::Int(1),
            Value::Int(2)
        ])
    );
}

/* def throttling_swap(d: list, e: int):
"""Swap positions of the 0'th and e'th elements in-place."""
e = throttling_mod_func(d, e)
f = d[0]
d[0] = d[e]
d[e] = f */

fn throttling_swap(d: &mut Value, e: Value) -> Value {
    match (d, e) {
        (Value::List(d), Value::Int(e)) => {
            let e = throttling_mod_func(d, e);
            let f = d[0].clone();
            d[0] = d[e as usize].clone();
            d[e as usize] = f;
        }
        _ => panic!("throttling_swap() expects a list"),
    }
    Value::Null
}

/*
def test_throttling_swap():
    a = [1, 2, 3, 4]
    cipher.throttling_swap(a, 3)
    assert a == [4, 2, 3, 1]
*/
#[test]
fn test_throttling_swap() {
    let mut a = Value::List(vec![
        Value::Int(1),
        Value::Int(2),
        Value::Int(3),
        Value::Int(4),
    ]);
    throttling_swap(&mut a, Value::Int(3));
    assert_eq!(
        a,
        Value::List(vec![
            Value::Int(4),
            Value::Int(2),
            Value::Int(3),
            Value::Int(1)
        ])
    );
}

/* def js_splice(arr: list, start: int, delete_count=None, *items):
"""Implementation of javascript's splice function.

:param list arr:
    Array to splice
:param int start:
    Index at which to start changing the array
:param int delete_count:
    Number of elements to delete from the array
:param *items:
    Items to add to the array

Reference: https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Array/splice  # noqa:E501
"""
# Special conditions for start value
try:
    if start > len(arr):
        start = len(arr)
    # If start is negative, count backwards from end
    if start < 0:
        start = len(arr) - start
except TypeError:
    # Non-integer start values are treated as 0 in js
    start = 0

# Special condition when delete_count is greater than remaining elements
if not delete_count or delete_count >= len(arr) - start:
    delete_count = len(arr) - start  # noqa: N806

deleted_elements = arr[start:start + delete_count]

# Splice appropriately.
new_arr = arr[:start] + list(items) + arr[start + delete_count:]

# Replace contents of input array
arr.clear()
for el in new_arr:
    arr.append(el)

return deleted_elements */

fn js_splice(arr: &mut Value, start: Value, delete_count: Value, items: Value) -> Value {
    match (arr, start, delete_count, items) {
        (Value::List(arr), Value::Int(start), Value::Int(delete_count), Value::List(items)) => {
            let start = if start > arr.len() as i32 {
                arr.len() as i32
            } else if start < 0 {
                arr.len() as i32 - start
            } else {
                start
            };
            let delete_count = if delete_count >= arr.len() as i32 - start {
                arr.len() as i32 - start
            } else {
                delete_count
            };
            let deleted_elements = arr[start as usize..(start + delete_count) as usize].to_vec();
            let new_arr = arr[..start as usize].to_vec();
            let new_arr = new_arr
                .into_iter()
                .chain(items.into_iter())
                .chain(arr[(start + delete_count) as usize..].iter().cloned())
                .collect();
            *arr = new_arr;
            Value::List(deleted_elements)
        }
        _ => panic!("js_splice() expects a list"),
    }
}

/*
def find_object_from_startpoint(html, start_point):
    """Parses input html to find the end of a JavaScript object.
    :param str html:
        HTML to be parsed for an object.
    :param int start_point:
        Index of where the object starts.
    :rtype dict:
    :returns:
        A dict created from parsing the object.
    """
    html = html[start_point:]
    if html[0] not in ['{','[']:
        raise HTMLParseError(f'Invalid start point. Start of HTML:\n{html[:20]}')

    # First letter MUST be a open brace, so we put that in the stack,
    # and skip the first character.
    stack = [html[0]]
    i = 1

    context_closers = {
        '{': '}',
        '[': ']',
        '"': '"'
    }

    while i < len(html):
        if len(stack) == 0:
            break
        curr_char = html[i]
        curr_context = stack[-1]

        # If we've reached a context closer, we can remove an element off the stack
        if curr_char == context_closers[curr_context]:
            stack.pop()
            i += 1
            continue

        # Strings require special context handling because they can contain
        #  context openers *and* closers
        if curr_context == '"':
            # If there's a backslash in a string, we skip a character
            if curr_char == '\\':
                i += 2
                continue
        else:
            # Non-string contexts are when we need to look for context openers.
            if curr_char in context_closers.keys():
                stack.append(curr_char)

        i += 1

    full_obj = html[:i]
    return full_obj  # noqa: R504

*/

fn find_object(html: &str) -> String {
    if !html.starts_with('{') && !html.starts_with('[') {
        panic!("Invalid start point. Start of HTML:\n{}", &html[..20]);
    }
    let mut html = html.chars();

    let first = html.next().unwrap();

    let mut stack = vec![first];

    let context_closers = vec![('}', '{'), (']', '['), ('"', '"')];

    let mut string = String::new();

    string.push(first);

    while !stack.is_empty() {
        let current = html.next().unwrap();
        string.push(current);
        if stack.last() == Some(&'"') {
            if current == '\\' {
                string.push(html.next().unwrap());
            } else if current == '"' {
                stack.pop();
            }
        } else if context_closers.iter().any(|(_, c)| *c == current) {
            stack.push(current);
        } else if context_closers.iter().any(|(c, _)| *c == current) {
            stack.pop();
        }
    }

    string
}

fn find_object_until(html: &str, until: char) -> String {
    let mut html = html.chars();

    let mut stack = vec![];

    let context_closers = vec![('}', '{'), (']', '['), ('"', '"')];

    let mut string = String::new();

    loop {
        let current = if stack.is_empty() {
            if let Some(e) = html.next() {
                if e == until {
                    return string;
                } else {
                    e
                }
            } else {
                return string;
            }
        } else {
            html.next().unwrap()
        };
        string.push(current);
        if stack.last() == Some(&'"') {
            if current == '\\' {
                string.push(html.next().unwrap());
            } else if current == '"' {
                stack.pop();
            }
        } else if context_closers.iter().any(|(_, c)| *c == current) {
            stack.push(current);
        } else if context_closers.iter().any(|(c, _)| *c == current) {
            stack.pop();
        }
    }
}

#[test]
fn test_find() {}

/*
def throttling_array_split(js_array):
    """Parses the throttling array into a python list of strings.

    Expects input to begin with `[` and close with `]`.

    :param str js_array:
        The javascript array, as a string.
    :rtype: list:
    :returns:
        A list of strings representing splits on `,` in the throttling array.
    """
    results = []
    curr_substring = js_array[1:]

    comma_regex = re.compile(r",")
    func_regex = re.compile(r"function\([^)]*\)")

    while len(curr_substring) > 0:
        if curr_substring.startswith('function'):
            # Handle functions separately. These can contain commas
            match = func_regex.search(curr_substring)
            match_start, match_end = match.span()

            function_text = find_object_from_startpoint(curr_substring, match.span()[1])
            full_function_def = curr_substring[:match_end + len(function_text)]
            results.append(full_function_def)
            curr_substring = curr_substring[len(full_function_def) + 1:]
        else:
            match = comma_regex.search(curr_substring)

            # Try-catch to capture end of array
            try:
                match_start, match_end = match.span()
            except AttributeError:
                match_start = len(curr_substring) - 1
                match_end = match_start + 1

            curr_el = curr_substring[:match_start]
            results.append(curr_el)
            curr_substring = curr_substring[match_end:]

    return results
*/
fn throttling_array_split(js_array: &str) -> Vec<String> {
    let mut js_array = &js_array[1..js_array.len() - 1];

    let mut array = Vec::new();

    loop {
        js_array = js_array.trim();
        if js_array.is_empty() {
            break;
        }
        let k = find_object_until(js_array, ',');
        js_array = &js_array[k.len()..];
        if js_array.starts_with(',') {
            js_array = &js_array[1..];
        }
        array.push(k);
    }

    array
}

fn log_debug(msg: &str) {
    // Append msg to the log-debug.txt file
    let mut file = OpenOptions::new()
        .append(true)
        .create(true)
        .open("log-debug.txt")
        .unwrap();
    file.write_all(msg.as_bytes()).unwrap();
    file.write_all(&[b'\n']).unwrap();
}
