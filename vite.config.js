export default {
    build: {
        outDir: 'dist',
        sourcemap: true,
        rollupOptions: {
            output: {
                manualChunks: {
                    tensorflow: ['@tensorflow/tfjs'],
                    mobilenet: ['@tensorflow-models/mobilenet']
                }
            }
        }
    }
}