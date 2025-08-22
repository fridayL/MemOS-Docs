<script lang="ts">
import type { FlatPathProps } from '@/utils/openapi'

interface NavLink {
  title: string
  children: Record<string, unknown>[]
}
</script>

<script setup lang="ts">
const { paths } = useOpenApi()

const methodColor = {
  post: {
    default: 'bg-blue-400/20 dark:bg-blue-400/20 text-blue-700 dark:text-blue-400',
    active: 'bg-[#3064E3] text-[#FFFFFF]'
  },
  get: {
    default: 'bg-green-400/20 dark:bg-green-400/20 text-green-700 dark:text-green-400',
    active: 'bg-[#2AB673] text-[#FFFFFF]'
  },
  delete: {
    default: 'bg-red-400/20 dark:bg-red-400/20 text-red-700 dark:text-red-400',
    active: 'bg-[#CB3A32] text-[#FFFFFF]'
  },
  put: {
    default: 'bg-yellow-400/20 dark:bg-yellow-400/20 text-yellow-700 dark:text-yellow-400',
    active: 'bg-[#C28C30] text-[#FFFFFF]'
  }
}

const navigationData = computed(() => {
  const result: NavLink = {
    title: 'API Reference',
    children: []
  }
  result.children = paths.value.map((path: FlatPathProps) => {
    return {
      title: path.summary,
      path: path.routePath,
      method: path.method
    }
  })

  return [result]
})
</script>

<template>
  <UContentNavigation :navigation="navigationData">
    <template #link-leading="{ link, active }">
      <span
        v-if="link.method"
        class="px-1 py-0.5 rounded-md text-[0.55rem] leading-tight font-bold"
        :class="active ? methodColor[link.method].active : methodColor[link.method].default"
      >
        {{ link.method.toUpperCase() }}
      </span>
    </template>
  </UContentNavigation>
</template>
