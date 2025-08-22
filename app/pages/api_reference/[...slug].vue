<script setup lang="ts">
const { getOpenApi, getApiByRoute } = useOpenApi()
await getOpenApi()

const route = useRoute()
const apiData = computed(() => getApiByRoute(route))
</script>

<template>
  <div class="flex">
    <!-- 移动端头部 -->
    <div class="block lg:hidden">
      <div class="z-30 fixed lg:sticky top-0 w-full">
        <div class="flex items-center lg:px-12 h-16 min-w-0 px-4">
          <div class="h-full relative flex-1 flex items-center justify-between gap-x-4 min-w-0 border-b border-gray-500/5 dark:border-gray-300/[0.06] lg:border-none">
            <NuxtLink>
              <LogoPro class="w-auto h-6 shrink-0" />
            </NuxtLink>
            <USlideover side="left">
              <button>
                <UIcon
                  name="i-lucide-align-justify"
                  class="w-[20px] h-[20px] cursor-pointer align-middle"
                />
              </button>
              <template #content>
                <ApiNavigation class="p-6" />
              </template>
            </USlideover>
          </div>
        </div>
      </div>
    </div>
    <div class="hidden lg:flex fixed flex-col left-0 top-0 bottom-0 w-[19rem] border-r border-gray-200/70 dark:border-white/[0.07]">
      <div class="flex-1 overflow-y-auto px-7 py-6">
        <NuxtLink>
          <LogoPro class="w-auto h-6 shrink-0" />
        </NuxtLink>
        <ApiNavigation class="mt-6" />
      </div>
    </div>
    <div class="relative w-full lg:ml-[19rem] flex gap-x-8 min-h-screen pt-40 px-4 lg:pt-10 lg:pl-16 lg:pr-10">
      <ApiContent
        v-if="apiData"
        :api-data="apiData"
      />
      <ApiNotFound v-else />
    </div>
  </div>
</template>
