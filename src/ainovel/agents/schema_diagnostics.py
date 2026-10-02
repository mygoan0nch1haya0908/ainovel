"""Bounded validation metadata only: never retain input values or validator messages."""

SCHEMA_ERROR_LABELS = {
    'missing': '缺少必填字段', 'extra_forbidden': '出现约定之外的字段',
    'int_type': '必须是整数', 'int_parsing': '不能解析为整数',
    'string_type': '必须是文本', 'list_type': '必须是列表', 'dict_type': '必须是对象',
    'model_type': '必须是对象', 'literal_error': '固定格式标记不正确',
    'greater_than_equal': '低于允许的最小值', 'less_than_equal': '超过允许的最大值',
    'too_short': '列表长度不足', 'too_long': '列表长度过长',
    'string_too_short': '文本过短', 'string_too_long': '文本过长',
    'string_pattern_mismatch': '标识格式不正确',
    'value_error': '结构约束未通过（例如顺序、依赖或总章数）',
    'validation_error': '未通过结构校验',
}


def safe_schema_issues(issues, schema):
    fields = {'unknown_field'}
    def visit(value):
        if isinstance(value, dict):
            fields.update(value.get('properties', {}))
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    visit(schema)
    result = []
    if not isinstance(issues, (list, tuple)):
        return result
    for issue in issues[:8]:
        if not isinstance(issue, dict):
            continue
        location = issue.get('loc', [])
        if not isinstance(location, (list, tuple)):
            location = []
        loc = [part if type(part) is int and 0 <= part <= 500
               else part if isinstance(part, str) and part in fields else 'unknown_field'
               for part in location[:6]]
        code = issue.get('type')
        result.append({'loc': loc, 'type': code if isinstance(code, str) and code in SCHEMA_ERROR_LABELS else 'validation_error'})
    return result
