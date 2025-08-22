type MethodType = 'post' | 'get' | 'delete' | 'put'

// 参数信息
interface ParametersProp {
  name: string
  in: 'path' | 'query'
  required: boolean
  schema: Record<string, string>
}

export interface PropertyProps {
  type?: string
  anyOf?: { type: string }[]
  title: string
  description: string
  example?: string
  default?: string
}

export interface SchemaProps {
  properties: Record<string, PropertyProps>
  required?: string[]
  title: string
  type: string
}

export interface ContentProps {
  [contentType: string]: {
    schema?: {
      $ref?: string
    }
  }
}

// 请求体信息
export interface RequestProps {
  required: boolean
  content: ContentProps
}

// 请求响应信息
interface ResponseProps {
  [key: string]: {
    description: string
    content: ContentProps
  }
}

// OpenAPI path信息
interface PathProps {
  description: string
  operationId: string
  parameters?: ParametersProp[]
  requestBody?: RequestProps
  responses: Record<string, ResponseProps>
  summary: string
}

export interface PathsProps {
  [key: string]: PathProps
}

// 扁平化后的路径
export interface FlatPathProps extends PathProps {
  method: MethodType
  apiUrl: string
  routePath: string
}

// 解析 schema 引用
export function resolveSchemaRef(
  ref: string | undefined | null,
  schemas: Record<string, unknown> | undefined | null
): Record<string, unknown> | null {
  if (!ref || !schemas) return null
  const key = ref.split('/').pop() as string | undefined
  if (!key) return null
  return (schemas[key] as Record<string, unknown>) || null
}

// 扁平化 OpenAPI paths
export function flattenPaths(paths: Record<string, PathsProps>) {
  const results: FlatPathProps[] = []

  Object.entries(paths).forEach(([apiUrl, methods]) => {
    Object.entries(methods).forEach(([method, operation]) => {
      const routePath = operation.summary.split(' ').map(s => s.toLowerCase()).join('-')

      results.push({
        apiUrl,
        method: method as MethodType,
        routePath: `/api_reference/${routePath}`,
        ...operation
      })
    })
  })

  return results
}
