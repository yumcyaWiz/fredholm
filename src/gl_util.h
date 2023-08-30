#pragma once
#ifndef __CUDACC__
#include <filesystem>
#include <string>
#include <variant>
#include <vector>

#include "Shadinclude.hpp"
#include "glad/gl.h"
#include "glm/glm.hpp"
#include "glm/gtc/type_ptr.hpp"
#include "spdlog/spdlog.h"

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

class GLVertexArrayObject
{
   private:
    GLuint array;

   public:
    GLVertexArrayObject() { glCreateVertexArrays(1, &array); }

    GLVertexArrayObject(const GLVertexArrayObject& other) = delete;

    GLVertexArrayObject(GLVertexArrayObject&& other) : array(other.array)
    {
        other.array = 0;
    }

    ~GLVertexArrayObject() { release(); }

    GLVertexArrayObject& operator=(const GLVertexArrayObject& other) = delete;

    GLVertexArrayObject& operator=(GLVertexArrayObject&& other)
    {
        if (this != &other)
        {
            release();

            array = other.array;

            other.array = 0;
        }

        return *this;
    }

    void bindVertexBuffer(const GLBuffer& buffer, GLuint binding,
                          GLintptr offset, GLsizei stride) const
    {
        glVertexArrayVertexBuffer(array, binding, buffer.getName(), offset,
                                  stride);
    }

    void bindElementBuffer(const GLBuffer& buffer) const
    {
        glVertexArrayElementBuffer(array, buffer.getName());
    }

    void activateVertexAttribution(GLuint binding, GLuint attrib, GLint size,
                                   GLenum type, GLsizei offset) const
    {
        glEnableVertexArrayAttrib(array, attrib);
        glVertexArrayAttribBinding(array, attrib, binding);
        glVertexArrayAttribFormat(array, attrib, size, type, GL_FALSE, offset);
    }

    void activate() const { glBindVertexArray(array); }

    void deactivate() const { glBindVertexArray(0); }

    void release()
    {
        if (array) { glDeleteVertexArrays(1, &array); }
    }
};

class GLPipeline
{
   private:
    class Shader
    {
       private:
        GLuint program;

        static GLuint createShaderProgram(GLenum type,
                                          const std::filesystem::path& filepath)
        {
            const std::string shader_source = Shadinclude::load(filepath.generic_string());
            const char* shader_source_c = shader_source.c_str();
            GLuint program = glCreateShaderProgramv(type, 1, &shader_source_c);
            return program;
        }

        static void checkCompileError(GLuint program)
        {
            int success = 0;
            glGetProgramiv(program, GL_LINK_STATUS, &success);
            if (success == GL_FALSE)
            {
                spdlog::error("failed to link program {:x}", program);

                GLint logSize = 0;
                glGetProgramiv(program, GL_INFO_LOG_LENGTH, &logSize);
                std::vector<GLchar> errorLog(logSize);
                glGetProgramInfoLog(program, logSize, &logSize, &errorLog[0]);
                std::string errorLogStr(errorLog.begin(), errorLog.end());
                spdlog::error("{}", errorLogStr);
            }
        }

       public:
        Shader() : program(0) {}

        void extracted(GLenum& type, const std::filesystem::path& filepath);

        Shader(GLenum type, const std::filesystem::path& filepath)
        {
            program = createShaderProgram(type, filepath);
            checkCompileError(program);
        }

        ~Shader() { release(); }

        Shader(const Shader& other) = delete;

        Shader(Shader&& other) : program(other.program) { other.program = 0; }

        Shader& operator=(const Shader& other) = delete;

        Shader& operator=(Shader&& other)
        {
            if (this != &other)
            {
                release();

                program = other.program;

                other.program = 0;
            }

            return *this;
        }

        operator bool() const { return this->program != 0; }

        void release()
        {
            if (program) { glDeleteProgram(program); }
        }

        GLuint getProgram() const { return program; }

        void setUniform(
            const std::string& uniform_name,
            const std::variant<bool, GLint, GLuint, GLfloat, glm::vec2,
                               glm::vec3, glm::mat4>& value) const
        {
            // get location of uniform variable
            const GLint location =
                glGetUniformLocation(program, uniform_name.c_str());

            // set value
            struct Visitor
            {
                GLuint program;
                GLint location;
                Visitor(GLuint program, GLint location)
                    : program(program), location(location)
                {
                }

                void operator()(bool value)
                {
                    glProgramUniform1i(program, location, value);
                }
                void operator()(GLint value)
                {
                    glProgramUniform1i(program, location, value);
                }
                void operator()(GLuint value)
                {
                    glProgramUniform1ui(program, location, value);
                }
                void operator()(GLfloat value)
                {
                    glProgramUniform1f(program, location, value);
                }
                void operator()(const glm::vec2& value)
                {
                    glProgramUniform2fv(program, location, 1,
                                        glm::value_ptr(value));
                }
                void operator()(const glm::vec3& value)
                {
                    glProgramUniform3fv(program, location, 1,
                                        glm::value_ptr(value));
                }
                void operator()(const glm::mat4& value)
                {
                    glProgramUniformMatrix4fv(program, location, 1, GL_FALSE,
                                              glm::value_ptr(value));
                }
            };
            std::visit(Visitor{program, location}, value);
        }

        static Shader createVertexShader(const std::filesystem::path& filepath)
        {
            return Shader(GL_VERTEX_SHADER, filepath);
        }

        static Shader createFragmentShader(
            const std::filesystem::path& filepath)
        {
            return Shader(GL_FRAGMENT_SHADER, filepath);
        }

        static Shader createGeometryShader(
            const std::filesystem::path& filepath)
        {
            return Shader(GL_GEOMETRY_SHADER, filepath);
        }

        static Shader createComputeShader(const std::filesystem::path& filepath)
        {
            return Shader(GL_COMPUTE_SHADER, filepath);
        }
    };

    GLuint pipeline;

    Shader vertex_shader;
    Shader fragment_shader;
    Shader geometry_shader;
    Shader compute_shader;

    void release()
    {
        if (pipeline) { glDeleteProgramPipelines(1, &pipeline); }
    }

    void attachVertexShader(Shader&& shader)
    {
        vertex_shader = std::move(shader);
        glUseProgramStages(pipeline, GL_VERTEX_SHADER_BIT,
                           vertex_shader.getProgram());
    }

    void attachGeometryShader(Shader&& shader)
    {
        geometry_shader = std::move(shader);
        glUseProgramStages(pipeline, GL_GEOMETRY_SHADER_BIT,
                           geometry_shader.getProgram());
    }

    void attachFragmentShader(Shader&& shader)
    {
        fragment_shader = std::move(shader);
        glUseProgramStages(pipeline, GL_FRAGMENT_SHADER_BIT,
                           fragment_shader.getProgram());
    }

    void attachComputeShader(Shader&& shader)
    {
        compute_shader = std::move(shader);
        glUseProgramStages(pipeline, GL_COMPUTE_SHADER_BIT,
                           compute_shader.getProgram());
    }

   public:
    GLPipeline() { glCreateProgramPipelines(1, &pipeline); }

    GLPipeline(const GLPipeline& other) = delete;

    GLPipeline(GLPipeline&& other) : pipeline(other.pipeline)
    {
        other.pipeline = 0;
    }

    ~GLPipeline() { release(); }

    GLPipeline& operator=(const GLPipeline& other) = delete;

    GLPipeline& operator=(GLPipeline&& other)
    {
        if (this != &other)
        {
            release();

            pipeline = other.pipeline;

            other.pipeline = 0;
        }

        return *this;
    }

    void loadVertexShader(const std::filesystem::path& filepath)
    {
        attachVertexShader(Shader::createVertexShader(filepath));
    }

    void loadGeometryShader(const std::filesystem::path& filepath)
    {
        attachGeometryShader(Shader::createGeometryShader(filepath));
    }

    void loadFragmentShader(const std::filesystem::path& filepath)
    {
        attachFragmentShader(Shader::createFragmentShader(filepath));
    }

    void loadComputeShader(const std::filesystem::path& filepath)
    {
        attachComputeShader(Shader::createComputeShader(filepath));
    }

    void setUniform(const std::string& uniform_name,
                    const std::variant<bool, GLint, GLuint, GLfloat, glm::vec2,
                                       glm::vec3, glm::mat4>& value) const
    {
        if (vertex_shader) { vertex_shader.setUniform(uniform_name, value); }
        if (geometry_shader)
        {
            geometry_shader.setUniform(uniform_name, value);
        }
        if (fragment_shader)
        {
            fragment_shader.setUniform(uniform_name, value);
        }
        if (compute_shader) { compute_shader.setUniform(uniform_name, value); }
    }

    void activate() const { glBindProgramPipeline(pipeline); }

    void deactivate() const { glBindProgramPipeline(0); }
};

class GLQuad
{
   private:
    GLVertexArrayObject VAO;
    GLBuffer vertex_buffer;
    GLBuffer index_buffer;

   public:
    GLQuad()
    {
        // setup VBO
        // position and texcoords
        const std::vector<GLfloat> vertices = {
            -1.0f, -1.0f, 0.0f, 0.0f, 0.0f, 1.0f,  -1.0f, 0.0f, 1.0f, 0.0f,
            1.0f,  1.0f,  0.0f, 1.0f, 1.0f, -1.0f, 1.0f,  0.0f, 0.0f, 1.0f};
        vertex_buffer.setData(vertices.data(), vertices.size(), GL_STATIC_DRAW);

        // setup EBO
        const std::vector<GLuint> indices = {0, 1, 2, 2, 3, 0};
        index_buffer.setData(indices.data(), indices.size(), GL_STATIC_DRAW);

        // setup VAO
        VAO.bindVertexBuffer(vertex_buffer, 0, 0, 5 * sizeof(GLfloat));
        VAO.bindElementBuffer(index_buffer);

        // position
        VAO.activateVertexAttribution(0, 0, 3, GL_FLOAT, 0);

        // texcoords
        VAO.activateVertexAttribution(0, 1, 2, GL_FLOAT, 3 * sizeof(GLfloat));
    }

    void draw(const GLPipeline& pipeline) const
    {
        pipeline.activate();
        VAO.activate();
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
        VAO.deactivate();
        pipeline.deactivate();
    }
};

}  // namespace fredholm

#endif