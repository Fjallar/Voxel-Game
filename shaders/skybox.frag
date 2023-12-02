#version 330 core
out vec4 FragColor;

in vec3 TexCoords;

uniform samplerCube u_texture_0;
void main()
{
    FragColor = texture(u_texture_0, TexCoords);
}