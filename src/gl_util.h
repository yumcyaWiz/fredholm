#pragma once
#ifndef __CUDACC__

#include "glad/gl.h"

namespace fredholm
{

class GLBuffer
{
   private:
    GLuint buffer = 0;
    uint32_t size = 0;

    void release()
    {
        if (buffer != 0)
        {
            glDeleteBuffers(1, &buffer);
            buffer = 0;
            size = 0;
        }
    }

   public:
    GLBuffer() : buffer(0), size(0) { glCreateBuffers(1, &buffer); }

    GLBuffer(const GLBuffer& buffer) = delete;

    GLBuffer(GLBuffer&& other) : buffer(other.buffer), size(other.size)
    {
        other.buffer = 0;
        other.size = 0;
    }

    ~GLBuffer() { release(); }

    GLBuffer& operator=(const GLBuffer& buffer) = delete;

    GLBuffer& operator=(GLBuffer&& other)
    {
        if (this != &other)
        {
            release();
            buffer = other.buffer;
            size = other.size;
            other.buffer = 0;
            other.size = 0;
        }
        return *this;
    }

    GLuint getName() const { return buffer; }

    uint32_t getLength() const { return size; }

    template <typename T>
    void setData(const T* data, uint32_t n, GLenum usage)
    {
        if (buffer)
        {
            glNamedBufferData(this->buffer, sizeof(T) * n, data, usage);
            this->size = n;
        }
    }

    void bindToShaderStorageBuffer(GLuint binding_point_index) const
    {
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding_point_index, buffer);
    }
};

}  // namespace fredholm

#endif